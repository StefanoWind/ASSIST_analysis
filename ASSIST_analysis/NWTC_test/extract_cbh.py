# -*- coding: utf-8 -*-
'''
Extract spectra in regions of interest
'''
import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import yaml
import xarray as xr
import glob
import warnings
warnings.filterwarnings('ignore')

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
sdate='2022-05-10'#[%Y-%m-%d] start date
edate='2022-08-25'#[%Y-%m-%d] end date
download=True#download new files?
channel='awaken/nwtc.ceil.z01.b0'

#%% Initalization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_dap'])
from doe_dap_dl import DAP

#%% Main

if download:
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    a2e.setup_basic_auth(username=config['username'], password=config['password'])

    _filter = {
        'Dataset': channel,
        'date_time': {
            'between': [sdate.replace('-',''),edate.replace('-','')]
        },
    }
        
    os.makedirs(os.path.join(cd,'data',channel),exist_ok=True)
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel),replace=False)
    
#extract cbh
cbh=None
files=sorted(glob.glob(os.path.join(cd,'data',channel,'*nc')))
for f in files:
    Data=xr.open_dataset(f).sortby('time')
    Data['time']=np.datetime64('1970-01-01T00:00:00')+Data.time*np.timedelta64(1, 's')
    
    if Data['time'].values[-1]>np.datetime64(sdate+'T00:00:00') and Data['time'].values[0]<np.datetime64(edate+'T00:00:00'):
        cbh_sel=xr.DataArray(data=Data['cloud_data'].values[:,0],coords={'time':Data.time})
        cbh_sel=cbh_sel.where(cbh_sel>=0)
        
        if cbh is None:
            cbh=cbh_sel
        else:
            cbh=xr.concat([cbh,cbh_sel], dim='time')
            
        print(os.path.basename(f))
        
#%% Output
Output=xr.Dataset()
Output['cbh']=cbh
Output.to_netcdf(os.path.join(cd,'data',sdate.replace('-','')+'.'+edate.replace('-','')+'.cbh.nc'))