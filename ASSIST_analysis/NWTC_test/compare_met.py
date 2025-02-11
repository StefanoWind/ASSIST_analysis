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
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import yaml
import statsmodels.api as sm
from scipy.stats import norm
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

unit='ASSIST10'
sources={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c0/*nc',
         'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
         'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}
source_met='data/nwtc.m5.a0/*nc'

height_assist=1
var='met_temperature_rec'

#qc
max_gamma=1
max_rmsa=5
min_lwp=5#[g/m^1]
max_height=0.2#[km]

#stats
p_value=0.05#for CI

colors=['b','g','y','r']
#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

name_save=os.path.join(cd,f'data/{unit}_met_1m.nc')

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
    
    
    Data=xr.Dataset()
    Data['met_temperature']=Data_met['temperature']
    Data['met_temperature_rec']=Data_met['temperature_rec']
    Data['trp_temperature']=Data_trp['temperature']
    Data['trp_sigma_temperature']=Data_trp['sigma_temperature']
    
    Data.to_netcdf(name_save)
else:
    Data=xr.open_dataset(name_save)


height=Data.height.values
time=Data.time.values

tnum=np.float64(time)/10**9
hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600

Data['DT']=Data['trp_temperature']-Data[var]
Data['hour']=xr.DataArray(data=hour, coords={'time':Data.time})

bins_hour=np.arange(25)
Data_dav = Data.groupby_bins("hour", bins_hour).mean()

Data_dsd=Data.groupby_bins("hour", bins_hour).std()

dT_dz=Data[var].differentiate('height')

#%% Plots

plt.close('all')
fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,Data[var],'-k',alpha=0.25,markersize=1)
    plt.plot(time,Data[var].isel(height=i_h),'-k',markersize=3)
    plt.plot(time,Data['trp_temperature'].isel(height=i_h),'-r',markersize=3)
    plt.ylim([-5,30])
    plt.grid()

fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,Data['DT'].isel(height=i_h),'-k',markersize=3)
    plt.ylim([-3,3])
    plt.grid()


fig=plt.figure(figsize=(10,10))  
bins=np.arange(-5,5.1,0.01)
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    DT=Data['trp_temperature'].isel(height=i_h)-Data[var].isel(height=i_h)
    plt.hist(DT,bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=0,scale=Data['trp_sigma_temperature'].isel(height=i_h).mean()),'r')
    plt.plot(bins,norm.pdf(bins,loc=DT.mean(),scale=DT.std()),'k')
    plt.grid()


fig=plt.figure(figsize=(10,10))  
for i_h in range(len(height)):
    plt.plot(Data_dav.hour,Data_dav['DT'].isel(height=i_h),label=r'$z=$'+str(height[i_h])+' m',color=colors[i_h])
        
plt.xlabel('Hour (UTC)')
plt.ylabel(r'Mean of $\Delta T$ (TROPoe-met) [$^\circ$C]')
plt.legend()
plt.grid()

fig=plt.figure(figsize=(10,10))  
for i_h in range(len(height)):
    plt.plot(Data_dav.hour,Data_dsd['DT'].isel(height=i_h),label=r'$z=$'+str(height[i_h])+' m',color=colors[i_h])
    plt.plot(Data_dav.hour,Data_dav['trp_sigma_temperature'].isel(height=i_h),'--',color=colors[i_h])
    plt.plot(Data_dav.hour,Data['DT'].isel(height=i_h).std().values+np.zeros(len(bins_hour)-1),'-',color=colors[i_h])
        
plt.xlabel('Hour (UTC)')
plt.ylabel(r'St.dev. of $\Delta T$ (TROPoe-met) [$^\circ$C]')
plt.legend()
plt.grid()

x_plot=np.array([-0.5,0.5])
fig=plt.figure(figsize=(18,4))  
for i_h in range(len(height)):
    ax=plt.subplot(1,len(height),i_h+1)
    x=dT_dz.isel(height=i_h).values
    y=Data['DT'].isel(height=i_h).values
    if np.sum(~np.isnan(x+y))>2:
        plt.plot(dT_dz.isel(height=i_h).values,Data['DT'].isel(height=i_h).values,'.k',alpha=0.1)
        LF=np.polyfit(x[~np.isnan(x+y)],y[~np.isnan(x+y)],1)
        rho=utl.nancorrcoef(x,y)[0,1]
        plt.plot(x_plot,x_plot*LF[0]+LF[1],'r')
        plt.text(x_plot[0]+0.05,3,f'slope = {np.round(LF[0],1)} m \n corr = {np.round(rho,1)}')
        plt.xlim(x_plot)
        plt.ylim([-4,4])
        plt.xlabel(r'$\frac{\delta T}{\delta z}$')
        plt.grid()