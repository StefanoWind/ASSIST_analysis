# -*- coding: utf-8 -*-
"""
Extract all high-frequency time series
"""
import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import xarray as xr
import matplotlib
import yaml
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
var_sel=['temperature','waterVapor',
         'sigma_temperature','sigma_waterVapor',
         'vres_temperature','vres_temperature',
         'rmsa','gamma','qc','cbh']
max_height=200#[m]

if len(sys.argv)==1:
    unit='ASSIST11'
else:
    unit=sys.argv[0]

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#%% Main

#load tropoe data
files=glob.glob(config['sources_trp'][unit])
Data_trp=xr.open_mfdataset(files).sel(height=slice(0,max_height/1000))

#qc tropoe data
Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()#remove clouds with low lwp

qc_gamma=Data_trp['gamma']<=config['max_gamma']
qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
qc_cbh=Data_trp['height']<Data_trp['cbh']
qc=qc_gamma*qc_rmsa*qc_cbh
Data_trp['qc']=~qc+0
    
print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter')
print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter')
print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter')

Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+config['height_assist'])

Data_trp[var_sel].to_netcdf(os.path.join(cd,'data',f'tropoe.{unit}.nc'))
Data_trp.close()
        
#load met data
files=glob.glob(config['source_met_b0'])

Data_met=xr.open_mfdataset(files)

if "air_temp_rec" in Data_met.data_vars:
    Data_met=Data_met.rename({"air_temp":"temperature_abs"}).rename({"air_temp_rec":"temperature"})
    
#time interpolation
tnum_trp=(Data_trp.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
tnum_met=(Data_met.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
time_diff=tnum_met.interp(time=Data_trp.time,method='nearest')-tnum_trp
Data_met=Data_met.interp(time=Data_trp.time)
Data_met['time_diff']=time_diff

Data_met.to_netcdf(os.path.join(cd,'data',f'met.b0.{unit}.nc'))
Data_met.close()

