# -*- coding: utf-8 -*-
"""
Extract CBH from different sites, including ARM
"""

# -*- coding: utf-8 -*-
import os
cd=os.path.dirname(__file__)
import sys
import re
import warnings
from datetime import datetime
import numpy as np
from doe_dap_dl import DAP
import yaml
import glob
import xarray as xr
from ftplib import FTP

warnings.filterwarnings('ignore')

#%% Inputs

if len(sys.argv)==1:
    source_config=os.path.join(cd,'configs/config.yaml')
    sdate='20221001'
    edate='20221002'
    download=True #download new files?
    replace=False#replace existing files
else:
    source_config=sys.argv[1]
    sdate=sys.argv[2]
    edate=sys.argv[3]
    download=sys.argv[4]=="True"
    replace=sys.argv[5]=="True"

#%% Functions
def dates_from_files(files):
    '''
    Extract data from data filenames
    '''
    dates=[]
    for f in files:
        match = re.search( r"\b\d{8}\.\d{6}\b", os.path.basename(f))
        dates.append(match.group().split('.')[0])
    
    return dates

def save_cbh(file,replace=False):
    '''
    Save CBH in TROPoe format
    '''
    try:
        #load data
        Data=xr.open_dataset(file)
        
        #time info
        tnum=np.float64(Data['time'].astype('datetime64[s]').values)/10**9
        basetime=np.floor(tnum[0]/(24*3600))*24*3600
        time_offset=tnum-basetime
        
        #naming
        dir_save_cbh=os.path.join(os.path.dirname(file)[:-2]+'cbh')
        name_save=os.path.basename(dir_save_cbh)+'.'+datetime.utcfromtimestamp(basetime).strftime('%Y%m%d.%H%M%S')+'.nc'
        
        if os.path.isfile(os.path.join(dir_save_cbh,name_save))==False or replace:
            #exract cbh from ceilometer or lidar
            if 'cloud_data' in Data:
                cbh=np.float64(Data['cloud_data'].values[:,0])
            elif 'first_cbh' in Data:
                cbh=Data['first_cbh'].where(Data['qc_first_cbh']==0).values
            elif 'dl_cbh' in Data:
                cbh=Data['dl_cbh'].values
            
            #save cbh
            Output=xr.Dataset()
            Output['first_cbh']=xr.DataArray(data=np.int32(np.nan_to_num(cbh,nan=-9999)),
                                             coords={'time':np.int32(time_offset)},
                                             attrs={'description':'First cloud base height','units':'m'})
        
            Output['base_time']=np.int64(basetime)
            Output.attrs['comment']='created on '+datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')+' by stefano.letizia@nrel.gov'
            
            os.makedirs(dir_save_cbh,exist_ok=True)
            Output.to_netcdf(os.path.join(dir_save_cbh,name_save))
            return f'{os.path.basename(file)} created'
        else:
            return f'{os.path.basename(file)} already created, skipped'
    except:
        return f'{os.path.basename(file)} failed'
        
#%% Initialization
print("Running lidar profile reconstruction:")
print(f"source_config: {source_config}")
print(f"sdate: {sdate}")
print(f"edate: {edate}")
print(f"download: {download}")
print(f"replace: {replace}")

#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

if download==True:
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    a2e.setup_basic_auth(username=config['wdh']['username'], password=config['wdh']['password'])
    
os.makedirs(config['save_path'],exist_ok=True)
    
#%% Main
for c in config['channels']:
    channel=config['channels'][c]
    
    #WDH pipeline
    if 'awaken' in channel:
        #download
        dir_save=os.path.join(config['save_path'],channel)

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
            dir_save=os.path.join(os.path.join(config['save_path'],'awaken',dir_name))
            os.makedirs(dir_save,exist_ok=True)
            
            #scan local files
            local_files=[os.path.basename(f) for f in sorted(glob.glob(os.path.join(dir_save,'*nc')))]
            
            #download files from ftp
            for f in files:
                if len(f)>2:
                    date_file= datetime.strptime(dates_from_files([f])[0],'%Y%m%d')
                    if date_file>=datetime.strptime(sdate,'%Y%m%d') and date_file<=datetime.strptime(edate,'%Y%m%d'):
                        if f not in local_files: 
                            local_filename=os.path.join(os.path.join(dir_save,f))
                            with open(local_filename, "wb") as fid:
                                ftp.retrbinary(f"RETR {channel+'/'+f}", fid.write)
                                print(f'{f} transferred',flush=True)
                        else:
                            print(f'{f} already transferred, skipped',flush=True)
                
    #process
    files=sorted(glob.glob(os.path.join(dir_save,'*nc')))
    dates=dates_from_files(files)
    
    for d in dates:
        file_sel=glob.glob(os.path.join(dir_save,'*'+d+'*nc'))
        if len(file_sel)==1:
            output=save_cbh(file_sel[0],replace)
            print(output,flush=True)
        else:
            print(f'Zero or multiple files found on {d}',flush=True)
            
                
        
    