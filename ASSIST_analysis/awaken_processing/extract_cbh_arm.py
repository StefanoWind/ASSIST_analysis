# -*- coding: utf-8 -*-
"""
Extract CBH from arm data
"""

# -*- coding: utf-8 -*-
import os
cd=os.path.dirname(__file__)
import sys
import re
import pandas as pd
import warnings
from datetime import datetime
from datetime import timedelta
import numpy as np
import glob
import xarray as xr
import matplotlib.pyplot as plt
import json
from scipy import interpolate
from ftplib import FTP
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import socket
import getpass
import yaml

warnings.filterwarnings('ignore')

#%% Inputs

if len(sys.argv)==1:
    source_config=os.path.join(cd,'config/config.yaml')
    sdate='20230601'
    edate='20230607'
    download=True #download new files?
    replace=True#replace existing files
    delete=False#delete input files?
else:
    source_config=sys.argv[1]
    sdate=sys.argv[2]
    edate=sys.argv[3]
    download=sys.argv[5]=="True"
    replace=sys.argv[6]=="True"
    delete=sys.argv[7]=="True"

#%% Functions
def dates_from_files(files):
    dates=[]
    for f in files:
        match = re.search( r"\b\d{8}\.\d{6}\b", os.path.basename(f))
        dates.append(match.group().split('.')[0])
    
    return dates

def save_cbh(file):
    Data=xr.open_mfdataset(file,combine="nested",concat_dim="time")
    
    #time info
    tnum=np.float64(Data['time'].astype('datetime64[s]').values)/10**9
    basetime=np.floor(tnum[0]/(24*3600))*24*3600
    time_offset=tnum-basetime
    
    #exract cbh
    if 'cloud_data' in Data:
        cbh=np.float64(Data['cloud_data'].values[:,0])
    elif 'first_cbh' in Data:
        cbh=Data['first_cbh'].values
    elif 'dl_cbh' in Data:
        cbh=Data['dl_cbh'].values
    
    #save cbh
    Output=xr.Dataset()
    Output['first_cbh']=xr.DataArray(data=np.int32(np.nan_to_num(cbh,nan=-9999)),
                                     coords={'time':np.int32(time_offset)},
                                     attrs={'description':'First cloud base height','units':'m'})

    Output['base_time']=np.int64(basetime)
    Output.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
    
    dir_save_cbh=os.path.join(os.path.dirname(file)[:-2]+'cbh')
    name_save=os.path.basename(dir_save_cbh)+'.'+datetime.utcfromtimestamp(basetime).strftime('%Y%m%d.%H%M%S')+'.nc'
    
    os.makedirs(dir_save_cbh,exist_ok=True)
    Output.to_netcdf(os.path.join(dir_save_cbh,name_save))
    
#%% Initialization
print("Running lidar profile reconstruction:")
print(f"source_config: {source_config}")
print(f"sdate: {sdate}")
print(f"edate: {edate}")
print(f"download: {download}")
print(f"replace: {replace}")
print(f"delete: {delete}")

#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
sys.path.append(config['path_dap'])
from doe_dap_dl import DAP

if download==True:
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    a2e.setup_basic_auth(username=config['wdh']['username'], password=config['wdh']['password'])
    
#%% Main
for c in config['channels']:
    channel=config['channels'][c]
    
    #WDH pipeline
    if 'awaken' in channel:
        #download
        dir_save=os.path.join(cd,'data',channel)

        if download:

            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [sdate+'000000',edate+'000000']
                },
                'file_type': 'nc',
            }
            a2e.download_with_order(_filter,path=dir_save, replace=False)
         
    #ARM ftp pipeline
    elif 'letizias1' in channel:
       
        if download:
            ftp = FTP(config['arm']['ftp_server'])
            ftp.login()
            ftp.cwd(channel)
            files = sorted([f for f in ftp.nlst() if f not in [".", ".."]]) 
            dir_name='.'.join(files[0].split('.')[:2])
            
            #create local folder
            dir_save=os.path.join(os.path.join(cd,'data','awaken',dir_name))
            os.makedirs(dir_save,exist_ok=True)
            
            local_files=[os.path.basename(f) for f in glob.glob(os.path.join(dir_save,'*nc'))]
            
            for f in files:
                if len(f)>2:
                    match = re.search( r"\b\d{8}\.\d{6}\b", os.path.basename(f))
                    date_file= datetime.strptime(match.group().split('.')[0],'%Y%m%d')
                    if date_file>=datetime.strptime(sdate,'%Y%m%d') and date_file<=datetime.strptime(edate,'%Y%m%d'):
                        if f not in local_files: 
                            local_filename=os.path.join(os.path.join(dir_save,f))
                            with open(local_filename, "wb") as fid:
                                ftp.retrbinary(f"RETR {channel+'/'+f}", fid.write)
                                print(f'Transferred {f}')
                        else:
                            print(f'Skipped {f}')
                
    #process
    files=glob.glob(os.path.join(cd,'data',dir_save,'*nc'))
    dates=dates_from_files(files)
    
    for d in dates:
        file_sel=glob.glob(os.path.join(cd,'data',dir_save,'*'+d+'*nc'))[0]
        save_cbh(file_sel)
            
                
        
    