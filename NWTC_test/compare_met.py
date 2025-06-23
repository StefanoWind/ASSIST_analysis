# -*- coding: utf-8 -*-
"""
Compare tropoe retrievals to met tower data
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import yaml
from scipy.stats import norm
import matplotlib.dates as mdates
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

#user
unit='ASSIST11'#assist id

#user
var_trp='temperature'
var_met='temperature'#selected temperature variable in M5 data

#stats
p_value=0.05#for CI
max_height=0.2#[km]
bins_hour=np.arange(25)#[h] hour bins
max_mad=10#[K] maximum deviation form median over height
min_T=-10#[C] minimum temperature
 
#graphics
cmap = plt.get_cmap("viridis")

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    

#%% Main
Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.nc'))
Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.b0.{unit}.nc'))

#interpolation
Data_trp=Data_trp.interp(height=Data_met.height_therm)
    
#extract coords
height=Data_met.height_therm.values
time=Data_met.time.values

#T difference
diff=Data_trp[var_trp]-Data_met[var_met]

#hourly stats
tnum=np.float64(time)/10**9
hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600


#%% Plots

plt.close('all')

#time series of T
fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,Data_met[var_met],'-k',alpha=0.25)
    plt.plot(time,Data_met[var_met].isel(height_therm=i_h),'-k',label='Met')
    plt.plot(time,Data_trp[var_trp].isel(height_therm=i_h),'-r',label='TROPoe')
    plt.ylim([-5,30])
    plt.grid()
    plt.ylabel(r'$T$ [$^\circ$C]')
    if i_h==len(height)-1:
        plt.xlabel('Time (UTC)')
    plt.text(time[10],25,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
plt.legend()

#time series of DT
fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,diff.isel(height_therm=i_h),'-k',markersize=3,label='TROPoe-met')
    # plt.plot(time,Data['trp_temperature_bias'].isel(height=i_h),'r',label='Prior bias')
    plt.ylim([-3,3])
    plt.grid()
    plt.ylabel(r'$\Delta T$' +'\n (TROPoe-met)'+r'[$^\circ$C]')
    if i_h==len(height)-1:
        plt.xlabel('Time (UTC)')
    plt.text(time[10],2,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
plt.legend()

#histograms of DT
fig=plt.figure(figsize=(10,10))  
bins=np.arange(-5,5.1,0.05)
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.hist(diff.isel(height_therm=i_h),bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=0,scale=Data_trp["sigma_temperature"].isel(height_therm=i_h).mean()),'r',label='TROPoe')
    plt.plot(bins,norm.pdf(bins,loc=diff.isel(height_therm=i_h).mean(),
                               scale=diff.isel(height_therm=i_h).std()),'k',label='Met')
    plt.grid()
    if i_h==len(height)-1:
        plt.xlabel(r'$\Delta T$ (TROPoe-met) [$^\circ$C]')
    plt.ylim([0,3.5])
plt.legend()

# #linear regressions
# plt.figure(figsize=(18,4))
# for i_h in range(len(height)):
#     plt.subplot(1,len(height),i_h+1)
#     utl.plot_lin_fit(Data_met[var_met].isel(height_therm=i_h).values,
#                      Data_trp[var_trp].isel(height_therm=i_h).values)
#     plt.xlim([-5,30])
#     plt.ylim([-5,30])
#     plt.xlabel(r'$T$ (met) [$^\circ$C]')
#     if i_h==0:
#         plt.ylabel(r'$T$ (TROPoe) [$^\circ$C]')
#     plt.text(20,0,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
    
