# -*- coding: utf-8 -*-
"""
Compare tropoe retrievals to met station data
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
from scipy.stats import norm
from matplotlib.ticker import NullFormatter
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs

source_config=os.path.join(cd,'configs','config.yaml')

sites=['B','C1a','G']
sigma_met=0.25#[C] uncertainty of met measurements [NOAA, 2004]
height_met=2#[m a.g.l.]

#sonic data paths
sources_snc={'A2':os.path.join(cd,'data/sa2.sonic.z01.c0.20230101.20240101.nc'),
             'A5':os.path.join(cd,'data/sa5.sonic.z01.c0.20230101.20240101.nc')}
#stats
p_value=0.05#for CI
max_height=100#[m] maximum height
max_T=45#[C] max threshold of selected variable
min_T=-10#[C] min threshold of selected variable
max_time_diff=60#[s] maximum difference in time between met and TROPoe
 
#graphics
site_names={'B':'South','C1a':'Middle','G':'North'}
site_diff_names={'C1a-B':'Middle-South','G-B':'North-South','G-C1a':'North-Middle'}

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#read all temperature data
T_trp={}
sigma_trp={}
T_met={}
time_all={}
for s in sites:
    #read and align data
    Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{s}.nc'))
    Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.b0.{s}.nc'))
    
    Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner",exclude=["height"])
    
    #QC
    print(f"{int(np.sum(Data_trp.qc!=0))} points fail QC in TROPoe")
    Data_trp=Data_trp.where(Data_trp.qc==0)
    
    print(f"{int(np.sum(Data_met.time_diff>max_time_diff))} points fail max_time_diff")
    Data_met=Data_met.where(np.abs(Data_met.time_diff)<=max_time_diff)
    
    T_trp[s]=Data_trp.temperature.interp(height=height_met)
    sigma_trp[s]=Data_trp.sigma_temperature.interp(height=height_met)
    T_met[s]=Data_met.temperature
    
    T_trp[s]=T_trp[s].where(T_trp[s]>=min_T).where(T_trp[s]<=max_T)
    T_met[s]=T_met[s].where(T_met[s]>=min_T).where(T_met[s]<=max_T)
    
    time_all[s]=Data_trp.time.values

#calculate temperature differences    
ctr=0
diff_trp={}
diff_met={}
diff_sigma_trp={}
for s1 in sites:
    for s2 in sites[ctr+1:]:
        diff_trp[f'{s2}-{s1}']=T_trp[s2]-T_trp[s1]
        diff_met[f'{s2}-{s1}']=T_met[s2]-T_met[s1]
        diff_sigma_trp[f'{s2}-{s1}']=(sigma_trp[s1]**2+sigma_trp[s2]**2)**0.5
    ctr+=1

#%% Main


#%% Plots
plt.close('all')
#time series of T
fig=plt.figure(figsize=(18,10))
ctr=1
for s in sites:
    ax=plt.subplot(len(sites),1,ctr)
    plt.plot(time_all[s],T_met[s],'-k',label='Met')
    plt.plot(time_all[s],T_trp[s],'-r',label='TROPoe')
    plt.ylim([-10,45])
    plt.grid()
    plt.ylabel(r'$T$ [$^\circ$C]')
    if ctr==0:
        plt.xlabel('Time (UTC)')
    ctr+=1
    plt.title(s)
plt.legend()
plt.tight_layout()

#time series of DT
fig=plt.figure(figsize=(18,10))
ctr=1
for s in sites:
    ax=plt.subplot(len(sites),1,ctr)
    plt.plot(time_all[s],T_trp[s]-T_met[s],'-k')
    plt.ylim([-5,5])
    plt.grid()
    plt.ylabel(r'$\Delta T$ (TROPoe-met) [$^\circ$C]')
    if ctr==0:
        plt.xlabel('Time (UTC)')
    ctr+=1
    plt.title(s)
plt.tight_layout()

#time series of T
fig=plt.figure(figsize=(18,10))
ctr=1
for s in diff_trp.keys():
    ax=plt.subplot(len(diff_trp.keys()),1,ctr)
    plt.plot(diff_met[s].time,diff_met[s],'-k',label='Met')
    plt.plot(diff_trp[s].time,diff_trp[s],'-r',label='TROPoe')
    plt.ylim([-4,4])
    plt.grid()
    plt.ylabel(r'$\Delta T$ ('+s+r') [$^\circ$C]')
    if ctr==0:
        plt.xlabel('Time (UTC)')
    ctr+=1
    plt.title(s)
plt.legend()
plt.tight_layout()

#linear regression
matplotlib.rcParams['font.size'] = 14
bins=np.arange(-5,5.1,0.05)
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2,len(sites)+1,width_ratios=[1]*len(sites)+[0.05]) 
ctr=0
for s in sites:
    ax=fig.add_subplot(gs[0,ctr])
    if ctr==len(sites)-1:
        cax=fig.add_subplot(gs[0,ctr+1])
    else:
        cax=None
    utl.plot_lin_fit(T_met[s].values,
                     T_trp[s].values,ax=ax,cax=cax,bins=100,legend=ctr==0,limits=[0,100])
    
    ax.set_xlim([-10,45])
    ax.set_ylim([-10,45])
    ax.set_xticks([-10,0,10,20,30,40])
    ax.set_yticks([-10,0,10,20,30,40])
    ax.grid(True)
    ax.set_xlabel(r'$T$ (met) [$^\circ$C]')
    if ctr==0:
        ax.set_ylabel(r'$T$ (TROPoe) [$^\circ$C]')
        plt.legend(draggable=True)
    else:
        ax.set_yticklabels([])
    
  
    ax=fig.add_subplot(gs[1,ctr])
    
    plt.hist(T_trp[s]-T_met[s],bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=(T_trp[s]-T_met[s]).mean(),
                               scale=(T_trp[s]-T_met[s]).std()),'k',label='Data')
    plt.plot(bins,norm.pdf(bins,loc=0,scale=(sigma_trp[s].mean()**2+sigma_met**2)**0.5),'r',label='Theory')
    ax.fill_between(bins,norm.pdf(bins,loc=0,scale=(sigma_trp[s].min()**2+sigma_met**2)**0.5),
                         norm.pdf(bins,loc=0,scale=(sigma_trp[s].max()**2+sigma_met**2)**0.5),color='r',alpha=0.25)
    ax.set_yscale('log')
    plt.grid()
    if ctr==0:
        ax.set_ylabel('PDF')
        plt.legend(draggable=True)
    else:
        ax.yaxis.set_major_formatter(NullFormatter())
    
    plt.xlabel(r'$\Delta T$ (TROPoe-met) [$^\circ$C]')
    plt.xlim([-4,4])
    plt.ylim([0.01,10])
    ctr+=1    
    
#linear regression (differences)
matplotlib.rcParams['font.size'] = 14
bins=np.arange(-5,5.1,0.05)
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2,len(sites)+1,width_ratios=[1]*len(sites)+[0.05]) 
ctr=0
for s in diff_trp.keys():
    ax=fig.add_subplot(gs[0,ctr])
    if ctr==len(sites)-1:
        cax=fig.add_subplot(gs[0,ctr+1])
    else:
        cax=None
    utl.plot_lin_fit(diff_met[s].values,
                     diff_trp[s].values,ax=ax,cax=cax,bins=100,legend=ctr==0,limits=[0,100])
    
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    ax.set_xticks([-4,-2,0,2,4])
    ax.set_yticks([-4,-2,0,2,4])
    ax.grid(True)
    ax.set_xlabel(r'$\Delta T$ ('+site_diff_names[s]+r', met) [$^\circ$C]')
    if ctr==0:
        ax.set_ylabel(r'$\Delta T$ ('+site_diff_names[s]+r', TROPoe) [$^\circ$C]')
        plt.legend(draggable=True)
    else:
        ax.set_yticklabels([])
    
  
    ax=fig.add_subplot(gs[1,ctr])
    
    plt.hist(diff_trp[s]-diff_met[s],bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=(diff_trp[s]-diff_met[s].values).mean(),
                               scale=(diff_trp[s]-diff_met[s]).std()),'k',label='Data')
    plt.plot(bins,norm.pdf(bins,loc=0,scale=(diff_sigma_trp[s].mean()**2+2*sigma_met**2)**0.5),'r',label='Theory')
    ax.fill_between(bins,norm.pdf(bins,loc=0,scale=(diff_sigma_trp[s].min()**2+2*sigma_met**2)**0.5),
                         norm.pdf(bins,loc=0,scale=(diff_sigma_trp[s].max()**2+2*sigma_met**2)**0.5),color='r',alpha=0.25)
    ax.set_yscale('log')
    plt.grid()
    if ctr==0:
        ax.set_ylabel('PDF')
        plt.legend(draggable=True)
    else:
        ax.yaxis.set_major_formatter(NullFormatter())
    
    ax.set_xlabel(r'$\Delta (\Delta T)$ ('+site_diff_names[s]+r', TROPoe-met) [$^\circ$C]')
    plt.xlim([-4,4])
    plt.ylim([0.01,10])
    ctr+=1    

