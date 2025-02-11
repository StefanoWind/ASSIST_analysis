# -*- coding: utf-8 -*-
"""
Estimate bias due prior
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
import pandas as pd
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
var='temperature_rec'

#qc
max_gamma=1
max_rmsa=5
min_lwp=5#[g/m^1]
max_height=0.2#[km]


#stats
p_value=0.05#for CI
bins_hour=np.arange(25)

#graphics
colors=['b','g','y','r']

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

#%% Main

# if not os.path.isfile(name_save):
    
#load met data
files=glob.glob(os.path.join(cd,source_met))
Data_met=xr.open_mfdataset(files).rename({"air_temp":"temperature"}).rename({"air_temp_rec":"temperature_rec"})
    
tnum=np.float64(Data_met.time)/10**9
hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600

syear=int(str(Data_met.time.values[0])[:4])
smonth=int(str(Data_met.time.values[0])[5:7])
eyear=int(str(Data_met.time.values[-1])[:4])
emonth=int(str(Data_met.time.values[-1])[5:7])
assert eyear==syear, "Dataset spans more than on year"
bins_month=np.array([],dtype='datetime64')
for m in np.arange(smonth,emonth+2):
    bins_month=np.append(bins_month,np.datetime64(f'{syear}-{m:02}-01T00:00:00'))

Data_met['hour']=xr.DataArray(data=hour, coords={'time':Data_met.time})
Data_met['tnum']=xr.DataArray(data=tnum, coords={'time':Data_met.time})

month=np.arange(smonth,emonth+1)
hour=utl.mid(bins_hour)
height=Data_met.height.values

T_dav=np.zeros((len(month),len(hour),len(height)))
ctr=0
for m1,m2 in zip(bins_month[:-1],bins_month[1:]):
    Data_sel=Data_met.sel(time=slice(m1,m2))
    avg= Data_sel.groupby_bins("hour", bins_hour).mean()
    T_dav[ctr,:,:]=avg[var].values
    ctr+=1
    
Data_met_dav=xr.Dataset()
Data_met_dav[var]=xr.DataArray(T_dav,coords={'month':month,'hour':hour,'height':height})

#%% Output
Data_met_dav.to_netcdf(os.path.join(cd,'data','met_prior2.nc'))

#%% Plots
for m in month:
    fig=plt.figure(figsize=(10,10)) 
    for i_h in range(len(Data_met_dav.height)):
        plt.plot(Data_met_dav.hour,Data_met_dav[var].sel(month=m).isel(height=i_h),label=r'$z=$'+str(Data_met_dav.height.values[i_h])+' m',color=colors[i_h])
    plt.xlabel('Hour (UTC)')
    plt.ylabel(r'$\langle T \rangle$ [$^\circ$C]')
    plt.grid()
    plt.title(f'Month = {m}')
    plt.legend()