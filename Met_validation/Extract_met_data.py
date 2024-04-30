# -*- coding: utf-8 -*-

import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import warnings
import glob
import pandas as pd

warnings.filterwarnings('ignore')

#%% Inputs
sources={10:'/srv/data/nfs/sletizia/awaken/sg.met.z01.b0/*nc',
          11:'/srv/data/nfs/sletizia/awaken/sb.met.z01.b0/*nc',
          12:'/srv/data/nfs/sletizia/awaken/sc1.met.z01.b0/*nc'}

#%% Initialization

#%% Main
for s in sources.keys():
    T=[]
    WS=[]
    RH=[]
    time=np.array([],dtype='datetime64')
    files=sorted(glob.glob(sources[s]))
    for f in files:
        print('Processing :'+f)
        Data=xr.open_dataset(f)

        time=np.append(time,Data['time'])
        T=np.append(T,Data['temperature'].where(Data['qc_temperature']==0))
        WS=np.append(WS,Data['average_wind_speed'].where(Data['qc_average_wind_speed']==0))
        RH=np.append(RH,Data['relative_humidity'].where(Data['qc_relative_humidity']==0))
        
    Output=pd.DataFrame()
    Output['temperature']=T
    Output['average_wind_speed']=WS
    Output['relative_humidity']=RH
    Output['Time']=time
    Output=Output.set_index('Time')
    Output.to_csv(os.path.join(cd,'data/Met_T_WS_RH_'+str(s)+'.csv'))