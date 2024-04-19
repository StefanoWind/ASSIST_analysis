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
source_sum='data/Summary_T_{ID}.csv'
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

#TROPoe data (read and interpolate on common time)
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
        rename[c]=c.replace('T_','T_'+str(ID)+'_').replace('vres_','vres_'+str(ID)+'_')
    Data_trp_int=Data_trp_int.rename(columns=rename)
    
    Data=pd.merge(Data,Data_trp_int,left_index=True,right_index=True)
    
print('Extracting met data')
for ID in IDs:
    Data_met=pd.read_csv(source_met.format(ID=ID))
    Data_met['Timenum']=np.array([SL.datenum(t,'%Y-%m-%d %H:%M:%S') for t in Data_met['Time UTC'].values])
    
    Data_met=Data_met.set_index('Timenum').drop(columns='Time UTC')

    Data_met_synch=SL.resample_flex_v2_2(Data_met, tnum1, tnum2, 'mean')
    
    Data_met_synch=Data_met_synch.rename(columns={'Temperature':'T_'+str(ID)+'_met'})
    Data=pd.merge(Data,Data_met_synch,left_index=True,right_index=True)
    
print('Extracting summary data')
for ID in IDs:
    Data_sum=pd.read_csv(source_sum.format(ID=ID))
    Data_sum['Timenum']=np.array([SL.datenum(t[:-10],'%Y-%m-%d %H:%M:%S') for t in Data_sum['Time'].values])
    
    Data_sum=Data_sum.set_index('Timenum').drop(columns='Time')

    Data_sum_synch=SL.resample_flex_v2_2(Data_sum, tnum1, tnum2, 'mean')
    
    Data_sum_synch=Data_sum_synch.rename(columns={'T_675':'T_675_'+str(ID)+'_sum'})
    Data_sum_synch=Data_sum_synch.rename(columns={'T_amb':'T_amb_'+str(ID)+'_sum'})
    Data_sum_synch=Data_sum_synch.rename(columns={'T_abb':'T_abb_'+str(ID)+'_sum'})
    Data=pd.merge(Data,Data_sum_synch,left_index=True,right_index=True)

#%% Output
Data['Time']=time
Data.set_index('Time').to_csv('data/All_T.csv')
