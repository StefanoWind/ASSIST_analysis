# -*- coding: utf-8 -*-
'''
Created on Fri Nov 22 15:36:52 2024

@author: sletizia
'''
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
import glob 
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source='data/rhod.assist.tropoe.z01.c0.20240714.011005.nc'

#%% Initialization
Data=xr.open_dataset(source)
date=str(Data.time.values[0])[:10]

#graphics
max_z=2#[Km]
max_gamma=1
max_rmsa=5
min_lwp=5#[g/m^1]
ticks={'temperature':np.arange(15,30.1),'waterVapor':np.arange(0,15.1),'sigma_temperature':np.arange(0,2.1,0.1)}
sel_time=[43,100]

#%% Main

#qc data
Data['cbh'][Data['lwp']<min_lwp]=Data['height'].max()

Data['temperature_qc']=Data['temperature'].where(Data['gamma']<=max_gamma).where(Data['rmsa']<=max_rmsa).where(Data['height']<=Data['cbh'])#[C]
Data['waterVapor_qc']=  Data['waterVapor'].where(Data['gamma']<=max_gamma).where(Data['rmsa']<=max_rmsa).where(Data['height']<=Data['cbh'])#[g/Kg]


#%% Plots

#plot profiles, vres, qc
plt.close('all')
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(5, 3, height_ratios=[5,5,1,1,1],width_ratios=[1,5,0.1])

Data=Data.resample(time=str(np.median(np.diff(Data['time']))/np.timedelta64(1,'m'))+'min').nearest(tolerance='1min')

ax0=fig.add_subplot(gs[0,0])
plt.plot(Data.vres_temperature.quantile(0.5,dim='time'),Data.height,'.-k')
plt.fill_betweenx(Data.height,Data.vres_temperature.quantile(.25,dim='time'),Data.vres_temperature.quantile(.75,dim='time'),color='k',alpha=0.25)
plt.grid()
ax0.set_ylabel(r'$z$ [Km]')
ax0.set_xlim([0,5])
ax0.set_ylim([0, max_z])

ax1=fig.add_subplot(gs[0,1])
CS=plt.contourf(Data.time,Data.height,Data.temperature_qc.T,ticks['temperature'],cmap='hot',extend='both')
plt.scatter(Data.time,Data.cbh,15,'w',edgecolor='k')

ax1.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax1.set_ylim([0, max_z])
ax1.grid()
ax1.xaxis.set_major_formatter(mdates.DateFormatter(''))
ax1.set_facecolor((0.9,0.9,0.9))

ax2=fig.add_subplot(gs[0,2])
cb = fig.colorbar(CS, cax=ax2)
cb.set_label(r'$T$ [$^\circ$C]')

ax3=fig.add_subplot(gs[1,0])
plt.plot(Data.vres_waterVapor.quantile(0.5,dim='time'),Data.height,'.-k')
plt.fill_betweenx(Data.height,Data.vres_waterVapor.quantile(.25,dim='time'),Data.vres_waterVapor.quantile(.75,dim='time'),color='k',alpha=0.25)
ax3.grid()
ax3.set_xlabel('Vertical \n resolution [Km]')
ax3.set_ylabel(r'$z$ [Km]')
ax3.set_xlim([0,5])
ax3.set_ylim([0, max_z])

ax4=fig.add_subplot(gs[1,1])
CS=plt.contourf(Data.time,Data.height,Data.waterVapor_qc.T,ticks['waterVapor'],cmap='GnBu',extend='max')
plt.scatter(Data.time,Data.cbh,15,'w',label='Cloud base height',edgecolor='k')

ax4.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax4.set_ylim([0,max_z])
ax4.grid()
ax4.xaxis.set_major_formatter(mdates.DateFormatter(''))
ax4.set_facecolor((0.9,0.9,0.9))

ax5=fig.add_subplot(gs[1,2])
cb = fig.colorbar(CS, cax=ax5, orientation='vertical')
cb.set_label(r'$r$ [g Kg$^{-1}$]')

ax6=fig.add_subplot(gs[2,1])
plt.plot(Data.time,Data.gamma,'.g')
plt.plot(Data.time,Data.gamma**0*max_gamma,'--g')
plt.ylabel(r'$\gamma$')
plt.grid()
ax6.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax6.xaxis.set_major_formatter(mdates.DateFormatter(''))

ax7=fig.add_subplot(gs[3,1])
plt.plot(Data.time,Data.rmsa,'.r')
plt.plot(Data.time,Data.rmsa**0*max_rmsa,'--r')
plt.ylabel('RMSA')
ax7.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax7.xaxis.set_major_formatter(mdates.DateFormatter(''))
plt.grid()

ax8=fig.add_subplot(gs[4,1])
plt.plot(Data.time,Data.lwp,'.b')
plt.plot(Data.time,Data.lwp**0*min_lwp,'--b')
plt.ylabel(r'LWP [g m$^{-2}$]')
plt.grid()
ax8.set_xlabel('Time (UTC)')
ax8.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax8.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


#plot uncertianties
fig=plt.figure(figsize=(18,8))
ax1=plt.subplot(2,1,1)

CS=plt.contourf(Data.time,Data.height,Data.sigma_temperature.T,ticks['sigma_temperature'],cmap='coolwarm',extend='max')
plt.scatter(Data.time,Data.cbh,15,'w',edgecolor='k')

ax1.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax1.set_ylim([0, max_z])
ax1.set_ylabel('$z$ [Km]')
ax1.grid()
ax1.xaxis.set_major_formatter(mdates.DateFormatter(''))
ax1.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax1.set_facecolor((0.9,0.9,0.9))
cb = fig.colorbar(CS)
cb.set_label(r'$\sigma(T)$ [$^\circ$C]')

ax2=plt.subplot(2,1,2)
CS=plt.contourf(Data.time,Data.height,Data.sigma_waterVapor.T,ticks['sigma_temperature'],cmap='coolwarm',extend='max')
plt.scatter(Data.time,Data.cbh,15,'w',edgecolor='k')

ax2.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax2.set_ylim([0, max_z])
ax2.grid()
ax2.set_xlabel('Time (UTC)')
ax2.set_ylabel('$z$ [Km]')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.set_xlim([datetime.strptime(date,'%Y-%m-%d'),datetime.strptime(date,'%Y-%m-%d')+timedelta(days=1)])
ax2.set_facecolor((0.9,0.9,0.9))
cb = fig.colorbar(CS)
cb.set_label(r'$\sigma(r)$ [g Kg$^{-1}$]')

ctr=1
fig=plt.figure(figsize=(18,8))
for s in sel_time:
    A=Data.Akernal.sel(time=Data.time[s]).values.T[:110,:110]
    cbh_sel=Data.cbh.sel(time=Data.time[s]).values
    plt.subplot(1,len(sel_time),ctr)
    plt.pcolor(A,vmin=-0.3,vmax=0.3,cmap='seismic')
    if cbh_sel<Data['height'].max():
        plt.scatter(np.arange(110)*0+np.argmin(np.abs(cbh_sel-Data.height.values)),np.arange(110),15,'w',edgecolor='k')
        plt.scatter(np.arange(110)*0+55+np.argmin(np.abs(cbh_sel-Data.height.values)),np.arange(110),15,'w',edgecolor='k')
            
    plt.plot(np.arange(110),np.arange(110)*0+55,'k')
    plt.plot(np.arange(110)*0+55,np.arange(110),'k')
    plt.xlim([0,110])
    plt.ylim([0,110])
    
    plt.xticks(np.arange(0,110,5),np.round(np.concatenate([Data.height,Data.height])[::5],1),rotation=45)
    plt.yticks(np.arange(0,110,5),np.round(np.concatenate([Data.height,Data.height])[::5],1),rotation=45)
    plt.xlabel('$z$ [Km]')
    plt.ylabel('$z$ [Km]')
    ctr+=1
