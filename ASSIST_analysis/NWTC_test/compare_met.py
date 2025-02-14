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
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

#dataset
unit='ASSIST10'
sources={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c2/*nc',
         'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
         'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}
source_met='data/nwtc.m5.a0/*nc'
height_assist=1#[m] height of TROPoe's first point

#user
var='met_temperature_rec'#selected temperature variable in M5 data

#stats
p_value=0.05#for CI
max_height=0.2#[km]
bins_hour=np.arange(25)#[h] hour bins

#graphics
cmap = plt.get_cmap("viridis")

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

name_save=os.path.join(cd,f'data/{unit}_met_bias.nc')

#%% Main

if not os.path.isfile(name_save):

    #load tropoe data
    files=glob.glob(os.path.join(cd,sources[unit]))
    Data_trp=xr.open_mfdataset(files).sel(height=slice(0,max_height))
    
    #qc tropoe data
    Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()#remove clouds with low lwp
    
    qc_gamma=Data_trp['gamma']<=config['max_gamma']
    qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
    qc_cbh=Data_trp['height']<=Data_trp['cbh']
    qc=qc_gamma*qc_rmsa*qc_cbh
    Data_trp['temperature_qc']=Data_trp['temperature'].where(qc)#filter temperature
    Data_trp['waterVapor_qc']=  Data_trp['waterVapor'].where(qc)#filter mixing ratio
        
    print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter')
    print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter')
    print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter')
    
    #load met data
    files=glob.glob(os.path.join(cd,source_met))
    Data_met=xr.open_mfdataset(files).rename({"air_temp":"temperature"}).rename({"air_temp_rec":"temperature_rec"})

    #interpolation
    Data_met=Data_met.interp(time=Data_trp.time)
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+height_assist)
    Data_trp=Data_trp.interp(height=Data_met.height)
    
    #save output (to save time at next run)
    Data=xr.Dataset()
    Data['met_temperature']=Data_met['temperature']
    Data['met_temperature_rec']=Data_met['temperature_rec']
    Data['trp_temperature']=Data_trp['temperature']
    Data['trp_temperature_bias']=Data_trp['bias']
    Data['trp_sigma_temperature']=Data_trp['sigma_temperature']
    Data['cbh']=Data_trp['cbh']
    
    Data.to_netcdf(name_save)
else:
    Data=xr.open_dataset(name_save)

#extract coords
height=Data.height.values
time=Data.time.values

#T difference
Data['DT']=Data['trp_temperature']-Data[var]

#hourly stats
tnum=np.float64(time)/10**9
hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600
Data['hour']=xr.DataArray(data=hour, coords={'time':Data.time})
Data_dav = Data.groupby_bins("hour", bins_hour).mean()
Data_dsd=Data.groupby_bins("hour", bins_hour).std()

#%% Plots

plt.close('all')

#time series of T
fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,Data[var],'-k',alpha=0.25)
    plt.plot(time,Data[var].isel(height=i_h),'-k',label='Met')
    plt.plot(time,Data['trp_temperature'].isel(height=i_h),'-r',label='TROPoe')
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
    plt.plot(time,Data['DT'].isel(height=i_h),'-k',markersize=3,label='TROPoe-met')
    plt.plot(time,Data['trp_temperature_bias'].isel(height=i_h),'r',label='Prior bias')
    plt.ylim([-3,3])
    plt.grid()
    plt.ylabel(r'$\Delta T$' +'\n (TROPoe-met)'+r'[$^\circ$C]')
    if i_h==len(height)-1:
        plt.xlabel('Time (UTC)')
    plt.text(time[10],2,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
plt.legend()

#histograms od DT
fig=plt.figure(figsize=(10,10))  
bins=np.arange(-5,5.1,0.05)
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    DT=Data['trp_temperature'].isel(height=i_h)-Data[var].isel(height=i_h)
    plt.hist(DT,bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=0,scale=Data['trp_sigma_temperature'].isel(height=i_h).mean()),'r',label='TROPoe')
    plt.plot(bins,norm.pdf(bins,loc=DT.mean(),scale=DT.std()),'k',label='Met')
    plt.grid()
    if i_h==len(height)-1:
        plt.xlabel(r'$\Delta T$ (TROpoe-met) [$^\circ$C]')
    plt.ylim([0,3.5])
plt.legend()

#hourly bias
colors=[cmap(val) for val in np.linspace(0, 1, len(height))]
fig=plt.figure(figsize=(10,10))  
for i_h in range(len(height)):
    plt.plot(Data_dav.hour,Data_dav['DT'].isel(height=i_h),label=r'$z=$'+str(height[i_h])+' m',color=colors[i_h])
    plt.plot(Data_dav.hour,Data_dav['trp_temperature_bias'].isel(height=i_h),'--',color=colors[i_h])
        
plt.xlabel('Hour (UTC)')
plt.ylabel(r'Mean of $\Delta T$ (TROPoe-met) [$^\circ$C]')
plt.legend()
plt.grid()

#hourly std
fig=plt.figure(figsize=(10,10))  
for i_h in range(len(height)):
    plt.plot(Data_dav.hour,Data_dsd['DT'].isel(height=i_h),label=r'$z=$'+str(height[i_h])+' m',color=colors[i_h])
    plt.plot(Data_dav.hour,Data_dav['trp_sigma_temperature'].isel(height=i_h),'--',color=colors[i_h])
    plt.plot(Data_dav.hour,Data['DT'].isel(height=i_h).std().values+np.zeros(len(bins_hour)-1),'-',color=colors[i_h])
        
plt.xlabel('Hour (UTC)')
plt.ylabel(r'St.dev. of $\Delta T$ (TROPoe-met) [$^\circ$C]')
plt.legend()
plt.grid()
