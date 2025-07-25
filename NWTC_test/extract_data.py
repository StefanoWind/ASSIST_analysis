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

sampling_rate=14#[s] sampling rate of ASSIST

if len(sys.argv)==1:
    unit='ASSIST11'
else:
    unit=sys.argv[1]

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#%% Main

#load tropoe data
files=glob.glob(config['sources_trp'][unit])
Data_trp=xr.open_mfdataset(files)

#qc tropoe data
Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()#remove clouds with low lwp

qc_gamma=Data_trp['gamma']<=config['max_gamma']
qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
qc_cbh=Data_trp['height']<Data_trp['cbh']
qc=qc_gamma*qc_rmsa*qc_cbh
Data_trp['qc']=~qc+0
    
print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter', flush=True)
print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter', flush=True)
print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter', flush=True)

Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+config['height_assist'])

Data_trp[var_sel].compute().to_netcdf(os.path.join(cd,'data',f'tropoe.{unit}.nc'))
Data_trp.close()

#time information for interpolation
time_trp=Data_trp.time.values
tnum_trp=(time_trp-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')

#load met data
files=glob.glob(config['source_met_a1'])
dates=np.unique(np.array([f'{os.path.basename(f).split(".")[3]}' for f in files]))

#process one day at a time
for date in dates:
    files_sel=glob.glob(config['source_met_a1'].replace('*',f'*{date}*'))
    Data_met=xr.open_mfdataset(files_sel)
    
    if "air_temp_rec" in Data_met.data_vars:
        Data_met=Data_met.rename({"air_temp":"temperature_abs"}).rename({"air_temp_rec":"temperature"})
        
    #sampling rate matching
    dt_met=np.median(np.diff(Data_met.time))/np.timedelta64(1,'s')
    Data_met_res=Data_met.rolling(time=int(sampling_rate/dt_met), center=True).mean()
        
    #time interpolation
    tnum_met=(Data_met_res.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
    time_sel=(tnum_trp>=tnum_met.values[0])*(tnum_trp<=tnum_met.values[-1])
    if np.sum(time_sel)>0:
        time_diff=tnum_met.interp(time=time_trp[time_sel],method='nearest')-tnum_trp[time_sel]
        Data_met_int=Data_met_res.interp(time=time_trp[time_sel])
        Data_met_int['time_diff']=time_diff
        
        #save temp file
        Data_met_int.compute().to_netcdf(os.path.join(cd,'data',f'{date}.met.a1.{unit}.temp.nc'))
        Data_met.close()
        print(f"{date} done", flush=True)
    else:
        print(f"Skipping {date}", flush=True)

#combine daily files
files=glob.glob(os.path.join(cd,'data',f'*.met.a1.{unit}.temp.nc'))
Data_met=xr.open_mfdataset(files)
Data_met.compute().to_netcdf(os.path.join(cd,'data',f'met.a1.{unit}.nc'))
Data_met.close()

for f in files:
    os.remove(f)
