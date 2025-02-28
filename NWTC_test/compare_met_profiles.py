# -*- coding: utf-8 -*-
"""
Compare tropoe retrievals to met tower data by making video of profiles
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import yaml
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

#dataset
unit='ASSIST10'
sources={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c0',
         'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0',
         'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0'}
source_met='data/nwtc.m5.a0'

#user
date='2022-04-23'
max_height=200#[m] maximum height

#%% Initialization

#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

files_trp=glob.glob(os.path.join(cd,sources[unit],'*'+date.replace('-','')+'*nc'))
files_met=glob.glob(os.path.join(cd,source_met,'*'+date.replace('-','')+'*nc'))

os.makedirs(os.path.join(cd,'figures',date),exist_ok=True)

#%% Main
Data_trp=xr.open_mfdataset(files_trp).sel(height=slice(0,max_height))

#qc tropoe data
Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()#remove clouds with low lwp

qc_gamma=Data_trp['gamma']<=config['max_gamma']
qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
qc_cbh=Data_trp['height']<=Data_trp['cbh']
qc=qc_gamma*qc_rmsa*qc_cbh
Data_trp['temperature_qc']=Data_trp['temperature'].where(qc)#filter temperature
Data_trp['waterVapor_qc']=  Data_trp['waterVapor'].where(qc)#filter mixing ratio
    
print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter')
print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter')
print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter')

Data_met=xr.open_mfdataset(files_met).rename({"air_temp":"temperature"}).rename({"air_temp_rec":"temperature_rec"})
Data_met=Data_met.interp(time=Data_trp.time)

time=Data_trp.time.values
for i in np.arange(len(time)):
    plt.figure()
    plt.plot(Data_met['temperature_rec'].isel(time=i),Data_met.height,'.-k',label='Met (diff. based)')
    plt.plot(Data_met['temperature'].isel(time=i),Data_met.height,'.-b',label='Met')
    plt.plot(Data_trp['temperature_qc'].isel(time=i),Data_trp.height*1000,'.-r',label='TROPoe')
    plt.fill_betweenx(Data_trp.height*1000,
                      Data_trp['temperature_qc'].isel(time=i)-Data_trp['sigma_temperature'].isel(time=i),
                      Data_trp['temperature_qc'].isel(time=i)+Data_trp['sigma_temperature'].isel(time=i),
                      color='r',alpha=0.25)
    
    plt.xlabel(r'$T$ [$^\circ$C]')
    plt.ylabel('$z$ [m]')
    plt.xlim([-5,35])
    plt.ylim([0,200])
    plt.title(str(time[i]).replace('T',' ')[:-6])
    if Data_trp['lwp'].isel(time=i).values>config['min_lwp']:
        plt.text(0,175,f'CBH={int(Data_trp.cbh.isel(time=i).values*1000)} m')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cd,'figures',date,f'{i:03}.png'))  
    plt.close()
    
utl.make_video(os.path.join(cd,'figures',date),os.path.join(cd,'figures',date+'.profiles.mp4'),10)