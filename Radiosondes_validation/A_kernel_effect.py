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
from scipy.integrate import cumtrapz
import warnings
import matplotlib
import pandas as pd
import glob 
from scipy.stats import binned_statistic

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 16

#%% Inputs
month='07'
source_prior='data/prior/Xa_Sa_datafile.sgp.55_levels.month_{month}.cdf'
source='data/All_T.csv'
source_sonde='data/sgpsondewnpnS6.b1/*2023{month}*cdf'
source_tropoe='data/assist-10/nreltropoe_10.c0.20230508.000015.nc'
timezone=-6#timezone

IDs=[11,12,10]

#graphics
site_names={10:'North',
            11:'South',
            12:'Middle'}

bin_hour=np.arange(-1.5,26,3)
height_met=2
timezone=-6

#%% Initialization
Data_tropoe=xr.open_dataset(os.path.join(cd,source_tropoe))
height=Data_tropoe.height.values*10**3
A=np.nanmean(Data_tropoe.Akernal.values,axis=0)[:len(height),:len(height)].T
# A=Data_tropoe.Akernal.values[0,:len(height),:len(height)]
# A=np.eye(len(height))
Prior=xr.open_dataset(os.path.join(cd,source_prior.format(month=month)))
T_mean_prior=Prior['mean_temperature'].values
T_std_prior=Prior['covariance_prior'].values[0,0]**0.5

files_sonde=glob.glob(os.path.join(cd,source_sonde.format(month=month)))

T_sonde=[]
T_sonde_smooth=[]
tnum_sonde=[]
hour_sonde=[]
T_met_eq=[]

#%% Main
for f in files_sonde:
    Data=xr.open_dataset(f)
    time=Data['time'].values+np.timedelta64(timezone,'h')
    asc=Data['asc'].values
    T=Data.tdry.values
    tnum=np.float64(time)/10**9
    height_sonde=cumtrapz(asc,tnum,initial=0)
    
    T_interp=np.interp(height,height_sonde,T)
    T_sonde=utl.vstack(T_sonde,T_interp)
    T_sonde_smooth=utl.vstack(T_sonde_smooth,np.matmul(A,(T_interp-T_mean_prior))+T_mean_prior)
    
    T_met_eq=np.append(T_met_eq,np.interp(height_met,height_sonde,T))
    
    tnum_sonde=np.append(tnum_sonde,np.nanmean(tnum))
    hour_sonde=np.append(hour_sonde,(np.nanmean(tnum)-utl.floor(np.nanmean(tnum),3600*24))/3600)
    print(f)
         
T_sonde_avg=np.zeros((len(bin_hour)-1,len(height)))
DT_avg=np.zeros((len(bin_hour)-1,len(height)))
for i_z in range(len(height)):
    T_sonde_avg[:,i_z]=binned_statistic(hour_sonde, T_sonde[:,i_z],statistic='median',bins=bin_hour)[0]
    # T_sonde_smooth_avg[:,i_z]=binned_statistic(hour_sonde, T_sonde_smooth[:,i_z],statistic='mean',bins=bin_hour)[0]
    DT_avg[:,i_z]=binned_statistic(hour_sonde, T_sonde_smooth[:,i_z]-T_met_eq,statistic='median',bins=bin_hour)[0]
    
    
    
#%% Plots
plt.figure()
plt.plot(hour_sonde,T_sonde_smooth[:,0]-T_met_eq,'.k')
plt.plot(utl.mid(bin_hour),DT_avg[:,0],'k')
plt.plot(hour_sonde,T_sonde_smooth[:,1]-T_met_eq,'.b')
plt.plot(utl.mid(bin_hour),DT_avg[:,1],'b')

plt.figure()
i=10
plt.plot(T_sonde[i,:],height)
plt.plot(T_sonde_smooth[i,:],height)


# plt.figure()
# i=10
# plt.plot(T_test,height)
# plt.plot(np.matmul(A,(T_test-T_mean_prior))+T_mean_prior,height)