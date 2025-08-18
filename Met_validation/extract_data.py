# -*- coding: utf-8 -*-

import os
cd=os.path.dirname(__file__)
import sys
import yaml
import numpy as np
import xarray as xr
import warnings
import glob

warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    path_config=os.path.join(cd,'configs/config.yaml') #config path
else:
    path_config=os.path.join(cd,'config',sys.argv[1])#config path
    
var_sel=['temperature','waterVapor',
         'sigma_temperature','sigma_waterVapor',
         'vres_temperature','vres_temperature',
         'rmsa','gamma','qc','cbh']

sampling_rate=14#[s] sampling rate of ASSIST
    
#%% Initialization
#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#%% Main
for s in config['channels_trp']:
    
    #load tropoe data
    files=glob.glob(os.path.join(config['path_trp'],config['channels_trp'][s],'*nc'))
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
    
    #fix height
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+config['height_assist'])
    
    #time information for interpolation
    time_trp=Data_trp.time.values
    tnum_trp=(time_trp-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
    
    Data_trp[var_sel].compute().to_netcdf(os.path.join(cd,'data',f'tropoe.{s}.nc'))
    Data_trp.close()
    
    #load met data
    files=glob.glob(os.path.join(config['path_data'],config['channels_met'][s],'*nc'))
    Data_met=xr.open_mfdataset(files)
    
    #shift to center time bin
    Data_met['time']=Data_met['time']-np.median(np.diff(Data_met['time']))/2
    
    #wind direction
    Data_met['U']=Data_met.average_wind_speed*np.cos(np.radians(270-Data_met.wind_direction))
    Data_met['V']=Data_met.average_wind_speed*np.sin(np.radians(270-Data_met.wind_direction))

    #time interpolation
    tnum_met=(Data_met.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
    time_diff=tnum_met.interp(time=time_trp,method='nearest')-tnum_trp
    Data_met_int=Data_met.interp(time=time_trp)
    Data_met_int['time_diff']=time_diff
    
    #reintroduce wind direcion
    Data_met_int['wd']=(270-np.degrees(np.arctan2(Data_met_int['V'],Data_met_int['U'])))%360
    
    #save temp file
    Data_met_int.compute().to_netcdf(os.path.join(cd,'data',f'met.b0.{s}.nc'))
    Data_met.close()
