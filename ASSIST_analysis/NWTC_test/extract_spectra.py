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
import re
warnings.filterwarnings('ignore')

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
sdate='2022-05-10'#[%Y-%m-%d] start date
edate='2022-08-25'#[%Y-%m-%d] end date
download=False#download new files?
channels=['awaken/nwtc.assist.z02.00',
          'awaken/nwtc.assist.z03.00']
time_increment=7#[days] chunk of files processed at a time

tropoe_bands= np.array([[612.0,618.0],
                        [624.0,660.0],
                        [674.0,713.0],
                        [713.0,722.0],
                        [538.0,588.0],
                        [793.0,804.0],
                        [860.1,864.0],
                        [872.2,877.5],
                        [898.2,905.4]])#cm[^-1]

max_time_diff=np.timedelta64(6,'s')#[s] maximum time difference between two IRS spectra

#%% Initalization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_dap'])
from doe_dap_dl import DAP

#selected wnum band
wnum_min=np.min(tropoe_bands)-10
wnum_max=np.max(tropoe_bands)+10

#%% Main

if download:
    a2e = DAP('a2e.energy.gov',confirm_downloads=False)
    a2e.setup_basic_auth(username=config['username'], password=config['password'])
    for channel in channels:
        _filter = {
            'Dataset': channel,
            'date_time': {
                'between': [sdate.replace('-',''),edate.replace('-','')]
            },
            'ext1':'assistcha',
        }
        
        os.makedirs(os.path.join(cd,'data',channel),exist_ok=True)
        a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel),replace=False)
        
#set up processing bins
time_file={}
files={}
for channel in channels:
    time_file[channel]=np.array([],dtype='datetime64')
    files[channel]=np.array(sorted(glob.glob(os.path.join(cd,'data',channel,'*cdf'))))
    for f in files[channel]:
        datestr = re.search(r'\d{8}\.\d{6}', f).group(0)
        datestr_np=f"{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}T{datestr[9:11]}:{datestr[11:13]}:{datestr[13:15]}"
        time_file[channel]=np.append(time_file[channel],np.datetime64(datestr_np))

time_bins=np.arange(np.datetime64(sdate+'T00:00:00'),
                    np.datetime64(edate+'T00:00:00')+np.timedelta64(time_increment, 'D'),
                    np.timedelta64(time_increment, 'D'))
    
#extract radiance
for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    rad={}
    for channel in channels:
        rad[channel]=None
        sel=(time_file[channel]>=t1)*(time_file[channel]<t2)
        files_sel=files[channel][sel]
        if len(files_sel)==0:
            break
        for f in files_sel:
            
            #load data
            Data=xr.open_dataset(f).sortby('time')
            Data['time']=np.datetime64('1970-01-01T00:00:00')+Data.base_time*np.timedelta64(1, 'ms')+Data.time*np.timedelta64(1, 's')
            
            #QC and selection
            rad_qc=Data['mean_rad'].where(np.abs(Data.sceneMirrorAngle)<0.1).where(Data.hatchOpen==1)
            rad_sel=rad_qc.sel(wnum=slice(wnum_min,wnum_max))
            
            #stacking
            if rad[channel] is None:
                rad[channel]=rad_sel
            else:
                rad[channel]=xr.concat([rad[channel],rad_sel], dim='time')
                
            print(os.path.basename(f))
            
    if len(files_sel)>0:
        #match times
        time1=rad[channels[0]].time.values
        time2=rad[channels[1]].time.values
        
        # #matrix time matching
        # time_diff0 = abs(time1[:, None] - time2[None, :])
        # i_synch0=np.argmin(time_diff0,axis=1)
        # min_time_diff0=np.min(time_diff0,axis=1)
        
        #vector time matching
        i=0
        i_synch=np.empty(len(time1), dtype=int)
        min_time_diff=np.empty(len(time1), dtype='timedelta64[ns]')
        for t1 in time1:
            time_diff = abs(t1 - time2)
            i_synch[i]=np.argmin(time_diff)
            min_time_diff[i]=time_diff[i_synch[i]]
            i+=1
        
        synch1=min_time_diff<=max_time_diff
        synch2=i_synch[synch1]
        
        time1_synch=time1[synch1]
        time2_synch=time2[synch2]
        
        #build common structure
        time_synch=time1_synch+(time2_synch-time1_synch)/2
        wnum_synch=rad[channels[0]].wnum.values
        
        rad_all=np.zeros((len(time_synch),len(wnum_synch),2))
        rad_all[:,:,0]=rad[channels[0]].values[synch1,:]
        rad_all[:,:,1]=rad[channels[1]].values[synch2,:]
        
        #Output
        Output=xr.Dataset()
        
        Output['rad']=xr.DataArray(data=rad_all,coords={'time':time_synch,'wnum':wnum_synch,'channel':channels})
        Output['time_diff']=xr.DataArray(data=(time2_synch-time1_synch),coords={'time':time_synch})
        
        Output.sortby('time').to_netcdf(os.path.join(cd,'data',
                                      str(np.min(time_synch))[:10].replace('-','')+\
                                  '.'+str(np.max(time_synch))[:10].replace('-','')+'.irs.nc'))
            