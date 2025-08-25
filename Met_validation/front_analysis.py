# -*- coding: utf-8 -*-
"""
Plot data during a frontal passage
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
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
path_config=os.path.join(cd,'configs/config.yaml') #config path
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')
sites_trp=['B','C1a','G']
sites_met=['A1','A2','A5','A7','B','C1a','G']

sources_trp={'B':'sb.assist.tropoe.z01.c0/sb.assist.tropoe.z01.c0.20230805.001005.nc',
             'C1a':'sc1.assist.tropoe.z01.c0/sc1.assist.tropoe.z01.c0.20230805.001005.nc',
             'G':'sg.assist.tropoe.z01.c0/sg.assist.tropoe.z01.c0.20230805.001005.nc'}

path_trp='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken'

sources_met={'A1':'sa1.met.z01.b0.20230805*nc',
             'A2':'sa2.met.z01.b0.20230805*nc',
             'A5':'sa5.met.z01.b0.20230805*nc',
             'A7':'sa7.met.z01.b0.20230805*nc',
             'B':'sb.met.z01.b0.20230805*nc',
             'C1a':'sc1.met.z01.b0.20230805*nc',
             'G':'sg.met.z01.b0.20230805*nc',}

path_met=os.path.join(cd,'data','front')

times=np.array([np.datetime64('2023-08-05T10:41:09'),
       np.datetime64('2023-08-05T10:51:25'),
       np.datetime64('2023-08-05T10:59:49'),
       np.datetime64('2023-08-05T11:09:58'),
       np.datetime64('2023-08-05T11:18:22'),
       np.datetime64('2023-08-05T11:31:00')])


max_gamma=3

T_min=24
T_max=28

#%% Initialization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
Map=xr.open_dataset(source_layout,group='ground_sites')
    
T_trp={}
for s in sites_trp:
    
    Data_trp=xr.open_dataset(os.path.join(path_trp,sources_trp[s]))
    
    #qc tropoe data
    Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()#remove clouds with low lwp
    
    qc_gamma=Data_trp['gamma']<=max_gamma
    qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
    qc_cbh=Data_trp['height']<Data_trp['cbh']
    qc=qc_gamma*qc_rmsa*qc_cbh
    Data_trp['qc']=~qc+0
        
    print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter', flush=True)
    print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter', flush=True)
    print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter', flush=True)
    
    #fix height
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+config['height_assist'])
     
    T_trp[s]=Data_trp.temperature.where(Data_trp.qc==0)
    
    Data_trp.close()
    
T_met={}
for s in sites_met:
    files=glob.glob(os.path.join(path_met,sources_met[s]))
    Data_met=xr.open_mfdataset(files)
    
    T_met[s]=Data_met.temperature.where(Data_met.qc_temperature==0)
    
    Data_met.close()
    
#%% Main
T_trp_int={}
for s in sites_trp:
    T_trp_int[s]=T_trp[s].interp(height=2,time=times)
    
T_met_int={}
for s in sites_met:
    T_met_int[s]=T_met[s].interp(time=times)
    

#%% Plots
plt.close("all")
fig=plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(1,len(times)+1,width_ratios=[1]*len(times)+[0.05]) 
for i in range(len(times)):
    ax=fig.add_subplot(gs[i])
    for s in sites_met:
        sc=plt.scatter(Map.x_utm.sel(site=s)-Map.x_utm.sel(site='C1a'),Map.y_utm.sel(site=s)-Map.y_utm.sel(site='C1a'),
                    c=T_met_int[s].isel(time=i),s=50,edgecolor='k',cmap='hot',vmin=T_min,vmax=T_max)
    if i>0:
        ax.set_yticklabels([])
        
    plt.grid()
    
cax=fig.add_subplot(gs[0,i+1])
plt.colorbar(sc,cax,label=r'$T$ [$^\circ$C]')

fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(len(sites_trp),2,width_ratios=[1,0.05]) 
ctr=0
for s in sites_trp:
    ax=fig.add_subplot(gs[ctr,0])
    cf=plt.contourf(T_trp[s].time,T_trp[s].height,T_trp[s].T,np.arange(T_min,T_max+.1,0.2),cmap='hot',extend='both')
    plt.contour(T_trp[s].time,T_trp[s].height,T_trp[s].T,np.arange(T_min,T_max+.1,0.2),colors='k',linewidths=0.1,alpha=0.5,extend='both')
    
    if ctr==len(sites_trp)-1:
        plt.xlabel('Time (UTC)')
    else:
        ax.set_xticklabels([])
    plt.ylabel(r'$z$ [$^\circ$C]')
    plt.grid()
    sc=plt.scatter(T_met_int[s].time,np.zeros(len(times))+2,
                c=T_met_int[s],s=50,edgecolor='k',cmap='hot',vmin=T_min,vmax=T_max,zorder=10)
    
    plt.xlim([times[0]-np.timedelta64(600,'s'),times[-1]+np.timedelta64(600,'s')])
    plt.ylim([-20,500])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ctr+=1

cax = fig.add_subplot(gs[:, 1])
plt.colorbar(cf,cax,label=r'$T$ [$^\circ$C]')