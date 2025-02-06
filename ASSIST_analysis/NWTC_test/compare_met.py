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
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

unit='ASSIST10'
sources={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c0/*nc',
         'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
         'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}
source_met='data/nwtc.m5.a0/*nc'

#qc
max_gamma=1
max_rmsa=5
min_lwp=5#[g/m^1]
max_height=0.2#[km]

#stats
p_value=0.05#for CI

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl


#%% Main

if not os.path.isfile(os.path.join(cd,f'data/{unit}_met.nc')):

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
    Data_met=xr.open_mfdataset(files).rename({"air_temp_rec":"temperature"})

    #interpolation
    Data_met=Data_met.interp(time=Data_trp.time)
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000)
    Data_trp=Data_trp.interp(height=Data_met.height)
    
    Data=xr.Dataset()
    Data['met_temperature']=Data_met['temperature']
    Data['trp_temperature']=Data_trp['temperature']
    Data['trp_sigma_temperature']=Data_trp['sigma_temperature']
    
    Data.to_netcdf(os.path.join(cd,f'data/{unit}_met.nc'))
else:
    Data=xr.open_dataset(os.path.join(cd,f'data/{unit}_met.nc'))


height=Data.height.values
time=Data.time.values



#%% Plots
fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,Data['met_temperature'],'k',alpha=0.25)
    plt.plot(time,Data['met_temperature'].isel(height=i_h),'k')
    plt.plot(time,Data['trp_temperature'].isel(height=i_h),'-r')
    plt.ylim([0,30])
    
fig=plt.figure(figsize=(10,10))  
bins=np.arange(-5,5.1,0.01)
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    DT=Data['trp_temperature'].isel(height=i_h)-Data['met_temperature'].isel(height=i_h)
    plt.hist(DT,bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=0,scale=Data['trp_sigma_temperature'].isel(height=i_h).mean()),'r')
    plt.plot(bins,norm.pdf(bins,loc=DT.mean(),scale=DT.std()),'k')



        
  