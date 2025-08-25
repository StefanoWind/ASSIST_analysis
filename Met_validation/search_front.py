# -*- coding: utf-8 -*-
"""
Search a weather front
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cd,'../utils'))
import utils as utl
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import yaml
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import norm
from matplotlib.ticker import NullFormatter
import glob
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['savefig.dpi']=100
plt.close("all")

#%% Inputs

#dataset
source_config=os.path.join(cd,'configs','config.yaml')

sites=['B','C1a','G']
site_diff=['C1a-B','G-B','G-C1a']

#stats
p_value=0.05#for CI
perc_lim=[5,95]#percentile filter
max_T=45#[C] max threshold of selected variable
min_T=-10#[C] min threshold of selected variable
max_time_diff=60#[s] maximum difference in time between met and TROPoe
max_time_diff_L=30*60#[s] maximum difference in tme between sonic and TROPoe

#graphics
site_names={'B':'South','C1a':'Middle','G':'North'}
site_diff_names={'C1a-B':'Middle-South','G-B':'North-South','G-C1a':'North-Middle'}

#%% Functions
def dates_from_files(files):
    '''
    Extract data from data filenames
    '''
    import re
    dates=np.array([],dtype='datetime64')
    for f in files:
        match = re.search( r"\b\d{8}\.\d{6}\b", os.path.basename(f))
        datestr=match.group()
        dates=np.append(dates,np.datetime64(f'{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}T{datestr[9:11]}:{datestr[11:13]}:{datestr[13:15]}'))
    
    return dates

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

dates={}
os.makedirs(os.path.join(cd,'figures','all'),exist_ok=True)

#%% Main

#find dates
for c in config['channels_trp']:
    files=glob.glob(os.path.join(config['path_trp'],config['channels_trp'][c],'*nc'))
    dates[c]=dates_from_files(files)
    
# Convert first array to set
common_dates = set(next(iter(dates.values()))) 

# Intersect with the rest
for arr in list(dates.values())[1:]:
    common_dates &= set(arr)
    
for d in sorted(common_dates):
    Data={}
    date=str(d).replace('-','')[:8]
    plt.figure(figsize=(18,8))
    ctr=0
    for c in config['channels_trp']:
        file=glob.glob(os.path.join(config['path_trp'],config['channels_trp'][c],f'*{date}*nc'))[0]
        Data[c]=xr.open_dataset(file)
        Data[c]=Data[c].where(Data[c].height<1.1)
        ax=plt.subplot(3,2,ctr*2+1)
        plt.pcolor(Data[c].time,Data[c].height*1000,Data[c].temperature.T,vmin=Data[sites[0]].temperature.min(),vmax=Data['B'].temperature.max(),cmap='hot')
        plt.xlabel('Time (UTC)')
        plt.ylabel(r'$z$ [m]')
        plt.title(f'{c} on {date}')
        plt.ylim([0,1000])
        plt.colorbar(label=r'$T$ [$^\circ$C]')
        plt.grid()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M')) 
        ctr+=1
        
    ctr=1
    for s in site_diff:
        s1=s.split('-')[1]
        s2=s.split('-')[0]
        DT=Data[s2].temperature-Data[s1].temperature
        ax=plt.subplot(3,2,ctr*2)
        plt.pcolor(DT.time,DT.height*1000,DT.T,vmin=-2,vmax=2,cmap='seismic')
        plt.xlabel('Time (UTC)')
        plt.ylabel(r'$z$ [m]')
        plt.title(f'{s} on {date}')
        plt.ylim([0,1000])
        plt.colorbar(label=r'$\Delta T$ [$^\circ$C]')
        plt.grid()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M')) 
        ctr+=1
       
    plt.tight_layout()
    plt.savefig(os.path.join(cd,'figures','all',f'{date}.png'))
    plt.close()
       