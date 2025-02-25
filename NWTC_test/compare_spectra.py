# -*- coding: utf-8 -*-
"""
Compare spectra
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
import xarray as xr
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import yaml
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs

#user
source_config=os.path.join(cd,'configs','config.yaml')
source=os.path.join(cd,'data','20220510.20220824.irs_with_cbh.nc')
skip=1#how many timestep to skip

#constants
T_amb=20#[C] ambient reference temperature
k=1.380649*10**-23#[J/Kg] Boltzman's constant
h=6.62607015*10**-34#[J s] Plank's constant
c=299792458.0#[m/s] speed of light
tropoe_bands= np.array([[612.0,618.0],
                        [624.0,660.0],
                        [674.0,713.0],
                        [713.0,722.0],
                        [538.0,588.0],
                        [793.0,804.0],
                        [860.1,864.0],
                        [872.2,877.5],
                        [898.2,905.4]])#[cm^-1] basned used by TROPoes

#stats
max_err=0.01#fraction of ambient BB
perc_lim=[1,99]#[%] limits for outlier filter (acts at each wavenumber independetly)

#graphics
name={'awaken/nwtc.assist.z02.00':'11',
      'awaken/nwtc.assist.z03.00':'12'}

wnum_sel=[500,700,900]#[cm^-1] selectd for time series plot

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

Data=xr.open_dataset(source).isel(time=slice(None, None, skip))

#%% Main

rad_diff=Data.rad.sel(channel='awaken/nwtc.assist.z03.00')-Data.rad.sel(channel='awaken/nwtc.assist.z02.00')

bias=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)

estd=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==0),
                    kwargs={"func": np.nanstd,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)

bias_c=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==1),
                    kwargs={"func": np.nanmean,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)
estd_c=xr.apply_ufunc(utl.filt_stat,rad_diff.where(Data.cloud_flag==1),
                    kwargs={"func": np.nanstd,'perc_lim': perc_lim},
                    input_core_dims=[["time"]],  
                    vectorize=True)

#Plank's law
B=2*h*c**2*Data.wnum**3/(np.exp(h*c*Data.wnum*100/(k*(273.15+T_amb)))-1)*10**11

#%% Plots
plt.close('all')

#error stats vs wnum
fig=plt.figure(figsize=(18,8))
ax=fig.add_subplot(1,1,1)
for tb in tropoe_bands:
    rect = Rectangle((tb[0], -5), tb[1]-tb[0], 15, edgecolor='b', facecolor='b', linewidth=2,alpha=0.25)
    ax.add_patch(rect)
ax.fill_between(Data.wnum,max_err*B,-max_err*B, edgecolor='g', facecolor='g', linewidth=2,alpha=0.25)
plt.plot(Data.wnum,bias,'k',label='Mean (no clouds)')
plt.plot(Data.wnum,estd,'r',label='StDev (clouds)')
plt.plot(Data.wnum,bias_c,'--k',label='Mean (clouds)')
plt.plot(Data.wnum,estd_c,'--r',label='StDev (clouds)')
plt.xlabel(r'$\tilde{\nu}$ [cm $^{-1}$]')
plt.ylabel(r'$\Delta B$ ('+name['awaken/nwtc.assist.z03.00']+'-'+name['awaken/nwtc.assist.z02.00']+') [r.u.]')
plt.grid()
plt.legend()

#selected timeseries
for w in wnum_sel:
    fig=plt.figure(figsize=(18,8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.5,10,10])
    ax=fig.add_subplot(gs[0,0])
    plt.plot(Data.time,Data.cloud_flag.where(Data.cloud_flag==1)-1,'.k',markersize=1)
    plt.plot(Data.time,Data.cloud_flag.where(Data.cloud_flag==0),'.c',markersize=1)
    plt.xlim([Data.time.values[0],Data.time.values[-1]])
    plt.ylim([-0.1,0.1])
    plt.grid()
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.title(r'Cloud flag at $\tilde{\nu}='+str(w)+'$ cm$^{-1}$')
    
    ax=fig.add_subplot(gs[1,0])
    plt.plot(Data.time,Data.rad.sel(wnum=w,method='nearest').sel(channel='awaken/nwtc.assist.z02.00'),'k',
              linewidth=1,label=name['awaken/nwtc.assist.z02.00'])
    plt.plot(Data.time,Data.rad.sel(wnum=w,method='nearest').sel(channel='awaken/nwtc.assist.z03.00'),'r',
              linewidth=1,label=name['awaken/nwtc.assist.z03.00'])
    plt.grid()
    plt.xlim([Data.time.values[0],Data.time.values[-1]])
    plt.ylim([0,200])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.legend()
    plt.ylabel(r'$B$ [r.u.]')
    
    ax=fig.add_subplot(gs[2,0])
    plt.plot(rad_diff.time,rad_diff.sel(wnum=w,method='nearest'),'b',linewidth=1)
    plt.grid()
    plt.xlim([Data.time.values[0],Data.time.values[-1]])
    plt.ylim([-10,10])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel(r'$\Delta B$ ('+name['awaken/nwtc.assist.z03.00']+'-'+name['awaken/nwtc.assist.z02.00']+') [r.u.]')
