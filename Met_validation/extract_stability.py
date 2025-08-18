# -*- coding: utf-8 -*-
'''
Download lidar data

Inputs (both hard-coded and available as command line inputs in this order):
    t_start [%Y-%m-%d]: start date in UTC
    t_end [%Y-%m-%d]: end date in UTC
    download [bool]: whether to download new data
    path_config: path to general config file
'''
import os
cd=os.path.dirname(__file__)
import sys
import warnings
from datetime import datetime
from datetime import timedelta
import yaml
import glob
from doe_dap_dl import DAP
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    t_start='2023-08-01' #start date
    t_end='2023-08-31' #end date
    download=False #download new data?
    mfa=False
    path_config=os.path.join(cd,'configs/config.yaml') #config path
else:
    t_start=sys.argv[1] #start date
    t_end=sys.argv[2]  #end date
    download=sys.argv[3]=="True" #download new data?
    mfa=sys.argv[4]=="True" #use MFA on WDH
    path_config=sys.argv[5]#config path

    
#%% Initalization
print(f'Downloading lidar data from {t_start} to {t_end}: download={download}, MFA={mfa}, config={path_config}.')

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#DAP setup
if download:
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    if mfa:
        a2e.setup_two_factor_auth(username=config['username'], password=config['password'])
    else:
        a2e.setup_cert_auth(username=config['username'], password=config['password'])
    
    N_periods=(datetime.strptime(t_end, '%Y-%m-%d')-datetime.strptime(t_start, '%Y-%m-%d'))/timedelta(hours=config['time_increment'])
    time_bin=[datetime.strptime(t_start, '%Y-%m-%d') + timedelta(hours=config['time_increment']*x) for x in range(int(N_periods)+1)]

#%% Main

#download
if download==True:
    for t1,t2 in zip(time_bin[:-1],time_bin[1:]):
        for c in config['channels_snc']:
            channel=config['channels_snc'][c]
            save_path=os.path.join(config['path_data'],channel)
            
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between':  [datetime.strftime(t1, '%Y%m%d%H%M%S'),
                                 datetime.strftime(t2-timedelta(seconds=1), '%Y%m%d%H%M%S')]
                },
                'file_type': 'csv',
            }
            
            a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel), replace=False)

#extract data              
for c in config['channels_snc']:
    channel=config['channels_snc'][c]
    files=glob.glob(os.path.join(config['path_data'],channel,'*csv'))
    
    #read all files
    dfs=[]
    for f in files:
        df = pd.read_csv(f).iloc[1:,:]
        dfs.append(df)
    Data=pd.concat(dfs, ignore_index=True)
    
    #format time
    time=np.array([],dtype='datetime64')
    for y,m,d,H,M in zip(Data['year'].values,
                       Data['month'].values,
                       Data['day'].values,
                       Data['hour'].values,
                       Data['minute'].values):
        time=np.append(time,np.datetime64(f'{y}-{m}-{d}T{H}:{M}'))
    
    time+=np.median(np.diff(time))/2
    
    #get QCed variables
    Data.loc[Data["QC flag"] !='0', :] = np.nan
    L=pd.to_numeric(Data['Obukhov\'s length'], errors='coerce').values
    L[L>10**10]=np.nan
    wd=pd.to_numeric(Data['wind direction'], errors='coerce').values
    wd[wd>10**10]=np.nan

    #write output
    Output=xr.Dataset()
    Output['L']=xr.DataArray(L,coords={'time':time})
    Output['wd']=xr.DataArray(wd,coords={'time':time})
    
    Output.to_netcdf(os.path.join(config['path_data'],f'sonic.c0.{c}.nc'))