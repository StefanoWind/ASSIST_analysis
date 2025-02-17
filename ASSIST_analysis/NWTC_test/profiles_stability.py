# -*- coding: utf-8 -*-
"""
Cluster profiles by atmospheric stability
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import yaml
import pandas as pd
from scipy.stats import norm
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

source_stb=os.path.join(cd,'data/nwtc.m5.b1/*csv')
time_offset=np.timedelta64(300, 's')
height_sonic=[15,41,61,74,100,119]#[m] sonic heights

#user
unit='ASSIST10'
met='M5'

#dataset
sources_trp={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c2/*nc',
         'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
         'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}

sources_met={'M5':'data/nwtc.m5.a0/*nc',
             'M2':'data/nwtc.m2.a0/*nc'}

height_assist=1#[m] height of TROPoe's first point

#stats
max_height=0.2#[km]

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

#load L data
files=glob.glob(source_stb)
data_stb= pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

time=data_stb.iloc[:,0].values*np.timedelta64(1, 's')+np.datetime64('1970-01-01T00:00:00')+time_offset

L=np.zeros((len(time),len(height_sonic)))
for i_h in range(len(height_sonic)):
    L[:,i_h]=data_stb[f'MO_Length_Sonic_{height_sonic[i_h]}m (m)'].values
    
Data_stb=xr.Dataset()
Data_stb['L']=xr.DataArray(data=L,coords={'time':time,'height':height_sonic})


name_save_trp=os.path.join(cd,f'data/{unit}_all.nc')
name_save_met=os.path.join(cd,f'data/{met}_all.nc')


if not os.path.isfile(name_save_trp):

    #load tropoe data
    files=glob.glob(os.path.join(cd,sources_trp[unit]))
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
    
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+height_assist)
    
    Data_trp.to_netcdf(name_save_trp)
    Data_trp.close()
    
#load data
Data_trp=xr.open_dataset(name_save_trp)

if not os.path.isfile(name_save_met):
    #load met data
    files=glob.glob(os.path.join(cd,sources_met[met]))
    
    Data_met=xr.open_mfdataset(files)
    
    if "air_temp_rec" in Data_met.data_vars:
        Data_met=Data_met.rename({"air_temp":"temperature"}).rename({"air_temp_rec":"temperature_rec"})
        
    #interpolation
    Data_met=Data_met.interp(time=Data_trp.time)
    
    Data_met.to_netcdf(name_save_met)
    Data_met.close()

#load data
Data_met=xr.open_dataset(name_save_met)


Data_stb=Data_stb.interp(time=Data_trp.time)


T_met_avg1=Data_met['temperature_rec'].where(1/Data_stb['L'].isel(height=2)>1).mean(dim='time')
T_met_avg2=Data_met['temperature_rec'].where(1/Data_stb['L'].isel(height=2)<-5).mean(dim='time')

T_trp_avg1=Data_trp['temperature'].where(1/Data_stb['L'].isel(height=2)>1).mean(dim='time')
T_trp_avg2=Data_trp['temperature'].where(1/Data_stb['L'].isel(height=2)<-5).mean(dim='time')

plt.figure()
plt.plot(T_met_avg1,Data_met.height)
plt.plot(T_met_avg2,Data_met.height)
plt.plot(T_trp_avg1,Data_trp.height)
plt.plot(T_trp_avg2,Data_trp.height)
