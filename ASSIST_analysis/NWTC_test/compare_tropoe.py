# -*- coding: utf-8 -*-
"""
Compare tropoe retrievals
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
import xarray as xr
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import yaml
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import norm
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
units=['ASSIST11','ASSIST12']
sources={'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c1/*nc',
         'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c1/*nc'}

#qc
max_gamma=1
max_rmsa=5
min_lwp=5#[g/m^1]
max_height=3#[km]

#stats
p_value=0.05#for CI
sel_acf=[0,7,25]

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

#load data
if not os.path.isfile(os.path.join(cd,'data',f'DT{units[1]}-{units[0]}.nc')):
    Data={}
    for u in units:
        files=glob.glob(os.path.join(cd,sources[u]))
        Data[u]=xr.open_mfdataset(files).sel(height=slice(0,max_height))
        
        #qc data
        Data[u]['cbh'][(Data[u]['lwp']<min_lwp).compute()]=Data[u]['height'].max()#remove clouds with low lwp
        
        qc_gamma=Data[u]['gamma']<=max_gamma
        qc_rmsa=Data[u]['rmsa']<=max_rmsa
        qc_cbh=Data[u]['height']<=Data[u]['cbh']
        qc=qc_gamma*qc_rmsa*qc_cbh
        Data[u]['temperature_qc']=Data[u]['temperature'].where(qc)#filter temperature
        Data[u]['waterVapor_qc']=  Data[u]['waterVapor'].where(qc)#filter mixing ratio
            
        print(f'{u}: {np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter')
        print(f'{u}: {np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter')
        print(f'{u}: {np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter')
        
    print('Computing temperature difference')
    DT=(Data[units[1]].temperature_qc-Data[units[0]].temperature_qc).compute()
    print('Computing uncertainty on temperature difference')
    sigmaDT=((Data[units[1]].sigma_temperature_n**2+Data[units[0]].sigma_temperature_n**2)**0.5).compute()
    
    Diff=xr.Dataset()
    Diff['DT']=DT
    Diff['sigmaDT']=sigmaDT
    Diff.to_netcdf(os.path.join(cd,'data',f'DT{units[1]}-{units[0]}.nc'))
else:
    Diff=xr.open_dataset(os.path.join(cd,'data',f'DT{units[1]}-{units[0]}.nc'))

bias_avg=xr.apply_ufunc(np.nanmean,Diff['DT'],
                    input_core_dims=[["time"]],  
                    vectorize=True)

bias_low=xr.apply_ufunc(utl.filt_BS_stat,Diff['DT'],
                    kwargs={"func": np.nanmean,'p_value':p_value*100/2,'perc_lim': [0,100]},
                    input_core_dims=[["time"]],  
                    vectorize=True)

bias_top=xr.apply_ufunc(utl.filt_BS_stat,Diff['DT'],
                    kwargs={"func": np.nanmean,'p_value':(1-p_value/2)*100,'perc_lim': [0,100]},
                    input_core_dims=[["time"]],  
                    vectorize=True)

estd_avg=xr.apply_ufunc(np.nanstd,Diff['DT'],
                    input_core_dims=[["time"]],  
                    vectorize=True)

estd_low=xr.apply_ufunc(utl.filt_BS_stat,Diff['DT'],
                    kwargs={"func": np.nanstd,'p_value':p_value*100/2,'perc_lim': [0,100]},
                    input_core_dims=[["time"]],  
                    vectorize=True)

estd_top=xr.apply_ufunc(utl.filt_BS_stat,Diff['DT'],
                    kwargs={"func": np.nanstd,'p_value':(1-p_value/2)*100,'perc_lim': [0,100]},
                    input_core_dims=[["time"]],  
                    vectorize=True)

kurt=xr.apply_ufunc(kurtosis,Diff['DT'],
                    input_core_dims=[["time"]],  
                    kwargs={'nan_policy':'omit'},
                    vectorize=True)

estd_th=xr.apply_ufunc(np.nanmean,Diff['sigmaDT'],
                    input_core_dims=[["time"]],  
                    vectorize=True)

nexc=xr.apply_ufunc(np.nansum,np.abs(Diff['DT'])>Diff['sigmaDT']*norm.ppf(1-p_value/2,loc=0,scale=1),
                    input_core_dims=[["time"]],  
                    vectorize=True)

nall=xr.apply_ufunc(np.nansum,~np.isnan(Diff['DT']+Diff['sigmaDT']),
                    input_core_dims=[["time"]],  
                    vectorize=True)


height=Diff.height.values*1000
time=Diff.time.values

time_int=np.arange(Diff.time.values[0],Diff.time.values[-1],np.nanmedian(np.diff(Diff.time)))
DT_interp=Diff['DT'].interp(time=time_int)

#%% Plots
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
ax0=fig.add_subplot(gs[0,0])
plt.plot(bias_avg,height,'k')
plt.fill_betweenx(height, bias_low,bias_top,color='k',alpha=0.25)
plt.xlim([-0.025,0.025])
plt.xlabel('Bias of $T$ [$^\circ$C]')
plt.ylabel('$z$ [m a.g.l.]')
plt.grid()

ax0=fig.add_subplot(gs[0,1])
plt.plot(estd_avg,height,'k')
plt.fill_betweenx(height, estd_low,estd_top,color='k',alpha=0.25)
plt.plot(estd_th,height,'r')
plt.xlabel('Error st.dev. of $T$ [$^\circ$C]')
plt.ylabel('$z$ [m a.g.l.]')
plt.grid()

ax0=fig.add_subplot(gs[0,2])
plt.plot(nexc/nall*100,height,'k')
plt.plot(height**0*p_value*100,height,'--r')
plt.xlabel(f'Probability of exceeding {(1-p_value)*100}% c.i. [%]')
plt.ylabel('$z$ [m a.g.l.]')
plt.grid()


plt.figure(figsize=(18,10))
ctr=1
for i_h in sel_acf:
    ax=plt.subplot(len(sel_acf),1,ctr)
    x=DT_interp.isel(height=i_h).values
    sm.graphics.tsa.plot_acf(x, lags=144*2,ax=ax,markersize=3,missing='conservative',adjusted=True)
    plt.gca().set_xscale('symlog')
    plt.xticks([0,1,6,144],['0','10 min','1 h','1 d'])
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.ylim([-0.1,1])
    plt.grid()
    plt.title(r'$z='+str(int(height[i_h]))+'$ m a.g.l.')
    
    ctr+=1
    
plt.figure(figsize=(18,10))
ctr=1
for i_h in sel_acf:
    ax=plt.subplot(len(sel_acf),1,ctr)
    plt.hist(Diff['DT'].isel(height=i_h),np.arange(-2,2.1,0.01),color='k',density=True)
    plt.plot(np.arange(-2,2.1,0.01),norm.pdf(np.arange(-2,2.1,0.01),loc=0,scale=Diff['sigmaDT'].isel(height=i_h).mean()),'-r')
    plt.plot(np.arange(-2,2.1,0.01),norm.pdf(np.arange(-2,2.1,0.01),loc=0,scale=Diff['DT'].isel(height=i_h).std()),'--r')
    plt.title(r'$z='+str(int(height[i_h]))+'$ m a.g.l.')
    plt.grid()
    ctr+=1