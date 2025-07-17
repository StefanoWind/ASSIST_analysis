# -*- coding: utf-8 -*-
"""
Extract all high-frequency M5 time series
"""
import os
cd=os.path.dirname(__file__)
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
var='temperature'
sampling_rate=14#[s] sampling rate of ASSIST

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#%% Main

#load met data
files=glob.glob(config['source_met_a1'])
dates=np.unique(np.array([f'{os.path.basename(f).split(".")[3]}' for f in files]))

#process one day at a time
for date in dates:
    files_sel=glob.glob(config['source_met_a1'].replace('*',f'*{date}*'))
    Data_met=xr.open_mfdataset(files_sel)
    
    if "air_temp_rec" in Data_met.data_vars:
        Data_met=Data_met.rename({"air_temp":"temperature_abs"}).rename({"air_temp_rec":"temperature"})
    
    Data_met=Data_met[var]
    
    #sampling rate matching
    dt_met=np.median(np.diff(Data_met.time))/np.timedelta64(1,'s')
    Data_met_res=Data_met.rolling(time=int(sampling_rate/dt_met), center=True).mean()
   
    #save temp file
    Data_met_res.compute().to_netcdf(os.path.join(cd,'data',f'{date}.met.a1.all.temp.nc'))
    Data_met.close()
    print(f"{date} done", flush=True)

#combine daily files
files=glob.glob(os.path.join(cd,'data','*.met.a1.all.temp.nc'))
Data_met=xr.open_mfdataset(files)
Data_met.compute().to_netcdf(os.path.join(cd,'data','met.a1.all.nc'))
Data_met.close()

for f in files:
    os.remove(f)
