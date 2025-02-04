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
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import yaml
import matplotlib.gridspec as gridspec

import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
units=['ASSIST11','ASSIST12']
sources={'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
         'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}

#qc
max_gamma=1
max_rmsa=5
min_lwp=5#[g/m^1]
max_height=3#[km]
perc_lim=[0,100]

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
        
        #smoothing error
        I=np.eye(len(Data[u].arb_dim1))
        Ss=np.zeros_like(Data[u].Sop)
        
        for it in range(len(Data[u].time)):
            A=Data[u]['Akernal'].isel(time=it).values.T
            Sa=Data[u].Sa.isel(time=it).values
            Ss[it,:,:]=(A-I)@Sa@(A-I).T
            print(it/len(Data[u].time))
        
        Data[u]['Ss']=xr.DataArray(data=Ss,coords={'time':Data[u].time,'arb_dim1':Data[u].arb_dim1,'arb_dim2':Data[u].arb_dim2})
        
        height=Data[units[0]].height.values

        sigma_temperature_n=np.zeros_like(Data[u].sigma_temperature)
        for it in range(len(Data[u].time)):
            sigma_temperature_n[it,:]=np.diag(Data[u]['Sop'].isel(time=it)-Data[u]['Ss'].isel(time=it))[:len(Data[u].height)]**0.5
            print(it/len(Data[u].time))
        Data[u]['sigma_temperature_n']=xr.DataArray(data=sigma_temperature_n,coords={'time':Data[u].time,'height':Data[u].height})
        
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

bias=xr.apply_ufunc(utl.filt_stat,Diff['DT'],
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)

estd=xr.apply_ufunc(utl.filt_stat,Diff['DT'],
                    kwargs={"func": np.nanstd,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)
estd_th=xr.apply_ufunc(utl.filt_stat,Diff['sigmaDT'],
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)


#%% Plots
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
ax0=fig.add_subplot(gs[0,0])
plt.plot(bias,bias.height,'k')

ax0=fig.add_subplot(gs[0,1])
plt.plot(estd,estd.height,'k')

plt.plot(estd_th,estd.height,'r')
