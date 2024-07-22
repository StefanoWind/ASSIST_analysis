# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:33:35 2024

@author: sletizia
"""

# -*- coding: utf-8 -*-
"""
Plot deily cycles of temperature, detrended temperatures and their differences
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
import glob 

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 16

#%% Inputs
month='07'
source_prior='data/Xa_Sa_datafile.sgp.55_levels.month_{month}.cdf'
source='data/All_T.csv'
timezone=-6#timezone

IDs=[11,12,10]

#graphics
site_names={10:'North',
            11:'South',
            12:'Middle'}

#%% Initialization
Data=pd.read_csv(os.path.join(cd,source))
Prior=xr.open_dataset(os.path.join(cd,source_prior.format(month=month)))

Data['Time']=np.array([utl.num_to_dt64(utl.datenum(t,'%Y-%m-%d %H:%M:%S')+timezone*3600) for t in Data['Time'].values])
Data=Data.set_index('Time')

Data['hour']=np.array([t.hour+t.minute/60 for t in Data.index])
Data['month']=np.array([t.month for t in Data.index])

Data=Data.where(Data['month']==int(month))

T_mean_prior=Prior['mean_temperature'].sel(height=0).values
T_std_prior=Prior['covariance_prior'].sel(height2=0).values[0,0]**0.5

#%% Main
T_0m_avg={}
T_0m_low={}
T_0m_top={}

for ID in IDs:
    T_0m_avg[ID],T_0m_low[ID],T_0m_top[ID]=utl.bins_unc(Data['hour'],(Data['T_{ID}_0.0m'.format(ID=ID)]-T_mean_prior)/T_std_prior,bins=np.arange(-0.5,24))
     
#%% Plots
fig=plt.figure(figsize=(18,5))
for ID in IDs:
    ax=plt.subplot(1,len(IDs),np.where(ID==np.array(IDs))[0][0]+1)

    plt.plot(np.arange(24),T_0m_avg[ID],'b')
    ax.fill_between(np.arange(24),T_0m_low[ID],T_0m_top[ID],color='b', alpha=0.25)
    plt.grid()
    plt.title(site_names[ID])
    plt.xticks([0,6,12,18],labels=['00','06','12','18'])
    plt.xlabel('Hour')
    plt.ylabel(r'$\frac{T-T_a}{\sqrt{S_a}}$ (TROPoe at 0 m)')
    plt.ylim([-2,2])
utl.remove_labels(fig)
    