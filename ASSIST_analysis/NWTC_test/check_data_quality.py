# -*- coding: utf-8 -*-
'''
Check quality of spectra
'''
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/utils')
import utils as utl
import numpy as np
import yaml
import xarray as xr
import glob
import warnings
import re
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
warnings.filterwarnings('ignore')

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
sdate='2022-05-15'#[%Y-%m-%d] start date
edate='2022-06-15'#[%Y-%m-%d] end date
download=True#download new files?
channel='awaken/nwtc.assist.z02.00'

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
        'ext1':'assistsummary',
    }
    
    os.makedirs(os.path.join(cd,'data',channel),exist_ok=True)
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel),replace=False)

#get fiel list
time_file=np.array([],dtype='datetime64')
files=np.array(sorted(glob.glob(os.path.join(cd,'data',channel,'*cdf'))))
for f in files:
    datestr = re.search(r'\d{8}\.\d{6}', f).group(0)
    datestr_np=f"{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}T{datestr[9:11]}:{datestr[11:13]}:{datestr[13:15]}"
    time_file=np.append(time_file,np.datetime64(datestr_np))

sel=(time_file>=np.datetime64(sdate+'T00:00:00'))*(time_file<np.datetime64(edate+'T00:00:00'))
files_sel=files[sel]

imag_rad990=[]
T990=[]
T680=[]
hatch=[]
time=np.array([],dtype='datetime64')
#extract quality info    
for f in files:
    
    #load data
    Data=xr.open_dataset(f).sortby('time')
    Data['time']=np.datetime64('1970-01-01T00:00:00')+Data.base_time+Data.time
    
    time=np.append(time,Data.time.values)
    imag_rad990=np.append(imag_rad990,Data.mean_imaginary_rad_985_990.values)
    T990=np.append(T990,Data.mean_Tb_985_990.values-273.15)
    T680=np.append(T680,Data.mean_Tb_675_680-273.15)
    hatch=np.append(hatch,Data.hatchOpen.values)
    print(f)
    
#%% Plots
plt.figure(figsize=(18,10))
plt.subplot(3,1,1)
plt.plot(time,T680,'.r',markersize=2)
plt.plot(time,T990,'.b',markersize=2)
plt.ylabel(r'$T_b$ at 990 cm$^{-1}$')
plt.title(channel)
date_format = mdates.DateFormatter('%Y%m%d') 
plt.gca().xaxis.set_major_formatter(date_format)
plt.grid()

plt.subplot(3,1,2)
plt.plot(time,imag_rad990,'.k',markersize=2)
plt.ylabel(r'imag$(B)$ at 990 cm$^{-1}$')

date_format = mdates.DateFormatter('%Y%m%d') 
plt.gca().xaxis.set_major_formatter(date_format)
plt.grid()
plt.gca().set_yscale('symlog')

plt.subplot(3,1,3)
plt.plot(time,hatch,'.k',markersize=2)
plt.ylabel(r'Hatch flag')
date_format = mdates.DateFormatter('%Y%m%d') 
plt.gca().xaxis.set_major_formatter(date_format)
plt.grid()
plt.xlabel('Time (UTC)')


plt.figure(figsize=(16,6))
ctr=1
for h in np.unique(hatch):
    plt.subplot(1,len(np.unique(hatch)),ctr)
    plt.hist(np.log10(np.abs(T680[hatch==h]-T990[hatch==h])),100,color='k')
    plt.xlim([-3,3])
    plt.title(f'{channel}: \n Hatch flag = {h}')
    plt.xlabel(r'log$_{10}(T_b(\tilde{\nu}=990)-T_b(\tilde{\nu}=680))$')
    plt.grid()
    ctr+=1

    
            
          