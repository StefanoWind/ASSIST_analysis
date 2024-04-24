# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:37:54 2024

@author: sletizia
"""

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions')
import myFunctions as SL

import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_trp='data/TROPoe_T_{ID}.csv'
source_met='data/Met_T_{ID}.csv'
source_inflow='data/20230101.000500-20240101.224500.awaken.glob.summary.csv'
start_time='2023-05-08 00:00:00.0'
end_time='2023-10-16 00:00:00.0'
time_res=30#min
IDs=[10,11,12]

max_DT=60#[s] maximum difference in time for interpolation of TROPoe data

#%% Initalization

#common time line
tnum=np.arange(SL.datenum(start_time),SL.datenum(end_time)+1,time_res*60)
time=[SL.num_to_dt64(t) for t in tnum]
tnum1=tnum-np.diff(tnum)[0]/2
tnum2=tnum+np.diff(tnum)[0]/2

Data=pd.DataFrame()
Data['Timenum']=tnum
Data=Data.set_index('Timenum')

print('Extracting TROPoe data')
for ID in IDs:
    Data_trp=pd.read_csv(source_trp.format(ID=ID))
    Data_trp['Timenum']=np.array([SL.datenum(t,'%Y-%m-%d %H:%M:%S') for t in Data_trp['Time'].values])
    
    Data_trp=Data_trp.set_index('Timenum').drop(columns='Time')
    Data_trp_int=SL.interp_df_v2(Data_trp, tnum, max_DT)
    
    rename={}
    for c in Data_trp_int.columns:
        rename[c]=c.replace('sigma_T_','sigma_T_'+str(ID)+'_').replace('T_','T_'+str(ID)+'_').replace('vres_','vres_'+str(ID)+'_')
    Data_trp_int=Data_trp_int.rename(columns=rename)
    
    Data=pd.merge(Data,Data_trp_int,left_index=True,right_index=True)
    
print('Extracting met data')
for ID in IDs:
    Data_met=pd.read_csv(source_met.format(ID=ID))
    Data_met['Timenum']=np.array([SL.datenum(t,'%Y-%m-%d %H:%M:%S') for t in Data_met['Time'].values])

    Data_met=Data_met.set_index('Timenum').drop(columns='Time')

    Data_met_synch=SL.resample_flex_v2_2(Data_met, tnum1, tnum2, 'mean')
    
    Data_met_synch=Data_met_synch.rename(columns={'Temperature':'T_'+str(ID)+'_met'})
    Data=pd.merge(Data,Data_met_synch,left_index=True,right_index=True)

print('Exctracting inflow data')
Data_inf=pd.read_csv(source_inflow)
Data_inf['Timenum']=np.array([SL.datenum(t,'%Y-%m-%d %H:%M:%S') for t in Data_inf['UTC Time'].values])
Data_inf=Data_inf.set_index('Timenum').drop(columns=['UTC Time','LLJ flag']).replace(-9999,np.nan)
Data_inf_synch=SL.resample_flex_v2_2(Data_inf, tnum1, tnum2, 'mean')

Data=pd.merge(Data,Data_inf_synch,left_index=True,right_index=True)

#%% Output
Data['Time']=time
Data.set_index('Time').to_csv('data/All_T.csv')
