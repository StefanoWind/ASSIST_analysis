# -*- coding: utf-8 -*-
'''
Check brightness temperature calculation
'''
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/utils')
import utils as utl
import numpy as np
import yaml
import xarray as xr
import glob
import warnings
import re
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
warnings.filterwarnings('ignore')

#%% Inputs
source_ch1=os.path.join(cd,'data/awaken/nwtc.assist.z01.00/assistcha.20220503.000046.cdf')
source_sum=os.path.join(cd,'data/awaken/nwtc.assist.z01.00/assistsummary.20220503.000046.cdf')

wnums1=[985,990]

wnums2=[675,680]

#%% Functions
def brightness_temp(rad,wnum):
    import numpy as np
    k=1.380649*10**-23#[J/Kg] Boltzman's constant
    h=6.62607015*10**-34#[J s] Plank's constant
    c=299792458.0#[m/s] speed of light

    Tb=100*h*c*wnum/k/np.log(2*10**11*c**2*h*wnum**3/rad+1)-273.15
    
    return Tb

#%% Initalization
Data_ch1=xr.open_dataset(source_ch1)
Data_sum=xr.open_dataset(source_sum)

time_ch1=Data_ch1.time*np.timedelta64(1,'s')+Data_ch1.base_time*np.timedelta64(1,'ms')+np.datetime64('1970-01-01T00:00:00')
time_sum=Data_sum.time+Data_sum.base_time+np.datetime64('1970-01-01T00:00:00')

#%% Main
Tb=brightness_temp(Data_ch1.mean_rad,Data_ch1.wnum)

T1=Tb.where(np.abs(Data_ch1.sceneMirrorAngle)<0.1).sel(wnum=slice(wnums1[0],wnums1[1])).mean(dim='wnum')
T2=Tb.where(np.abs(Data_ch1.sceneMirrorAngle)<0.1).sel(wnum=slice(wnums2[0],wnums2[1])).mean(dim='wnum')

#%% Plots
plt.figure()
plt.plot(time_sum,Data_sum[f'mean_Tb_{wnums1[0]}_{wnums1[1]}']-Data_sum[f'mean_Tb_{wnums2[0]}_{wnums2[1]}'],'k',label='summary')
plt.plot(time_ch1,T1-T2,'r',label='ch1')
plt.xlabel('Time')
plt.legend()
plt.grid()

plt.figure()
plt.plot(time_ch1,Data_ch1.hatchOpen,'.',label='ch1',markersize=10)
plt.plot(time_sum,Data_sum.hatchOpen,'.',label='summary')
