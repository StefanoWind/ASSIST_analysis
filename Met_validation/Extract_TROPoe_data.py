# -*- coding: utf-8 -*-

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions')
import myFunctions as SL

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import warnings
import matplotlib
import glob
import pandas as pd

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#dataset
sources='data/assist-{ID}/*000015.nc'
IDs=[10,11,12]
sel_gates=[0,1]#selected TROPoe gates for extraction

#QC
min_lwp=10#[g/m^2] minimum liquid water path for clouds
max_unc=2 #[C] maximum TROPoe uncertainty
max_gamma=5#max gamma of TROPoe
max_rmsr=5 #max rmsr of TROPoe retrieval

#%% Initialization
T={}
T_met={}
sigma_T={}
rmsr={}
gamma={}
cbh={}
lwp={}
vres={}
for ID in IDs:
    print('Reading ASSIST '+str(ID)+' data')
    ctr=0
    T[ID]=xr.DataArray()
    sigma_T[ID]=xr.DataArray()
    files=np.array(glob.glob(sources.format(ID=ID)))
    for f in files:
        Data=xr.open_dataset(f)
        if ctr==0:
            T[ID]=Data.temperature
            sigma_T[ID]=Data.sigma_temperature
            rmsr[ID]=Data.rmsr
            gamma[ID]=Data.gamma
            cbh[ID]=Data.cbh
            lwp[ID]=Data.lwp
            vres[ID]=Data.vres_temperature
           
        else:
            T[ID]=xr.concat([T[ID],Data.temperature],dim='time')
            sigma_T[ID]=xr.concat([sigma_T[ID],Data.sigma_temperature],dim='time')
            rmsr[ID]=xr.concat([rmsr[ID],Data.rmsr],dim='time')
            gamma[ID]=xr.concat([gamma[ID],Data.gamma],dim='time')
            cbh[ID]=xr.concat([cbh[ID],Data.cbh],dim='time')
            lwp[ID]=xr.concat([lwp[ID],Data.lwp],dim='time')
            vres[ID]=xr.concat([vres[ID],Data.vres_temperature],dim='time')
        ctr+=1


#%% Main
for ID in IDs:
    T[ID][rmsr[ID]>max_rmsr,:]=np.nan
    T[ID][gamma[ID]>max_gamma,:]=np.nan
    T[ID]=T[ID].where((T[ID].height>cbh[ID])*(lwp[ID]>min_lwp)==False)
    
    print('QC of ASSIST '+str(ID)+':')
    print(str(np.round(np.sum(rmsr[ID]>max_rmsr).values/len(rmsr[ID])*100,2))+'% excluded due to high RMSR')
    print(str(np.round(np.sum(gamma[ID]>max_gamma).values/len(gamma[ID])*100,2))+'% excluded due to high gamma')
    print(str(np.round(np.sum((T[ID].height>cbh[ID])*(lwp[ID]>min_lwp)).values/len(T[ID].values.ravel())*100,2))+'% excluded due to cloud presence')
    
    
#%% Output
for ID in IDs:
    Output=pd.DataFrame()
    for i_gate in sel_gates:
        z_sel=np.round(T[ID].height.values[i_gate]*1000,1)
        T_sel=T[ID].values[:,i_gate]
        vres_sel=vres[ID].values[:,i_gate]*1000
        Output['T_'+str(z_sel)+'m']=T_sel
        Output['vres_'+str(z_sel)+'m']=vres_sel
    Output['Time']=T[ID].time
    Output.set_index('Time').to_csv('data/TROPoe_T_'+str(ID)+'.csv')