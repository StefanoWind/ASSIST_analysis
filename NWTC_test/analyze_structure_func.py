# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:59:20 2025

@author: sletizia
"""

# -*- coding: utf-8 -*-
"""
Estimate impact of representativeness error
"""

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(cd,'../utils'))
import utils as utl
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import warnings
from scipy import stats
import glob
import yaml
import matplotlib
import matplotlib.dates as mdates
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source_met_sta=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats
source_met=os.path.join(cd,'data/nwtc/nwtc.m5.a1/temp/*nc')#source of 1-s met
max_ti=500
bin_hour=np.arange(0,25,2)
bin_wd=np.arange(0,361,30)

lags=[50,150,300]

#%% Initialization

#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#read met stats
files_sta=glob.glob(source_met_sta)
Data_met_sta=xr.open_mfdataset(files_sta)

files=glob.glob(source_met)

hour_avg=(bin_hour[:-1]+bin_hour[1:])/2
wd_avg=(bin_wd[:-1]+bin_wd[1:])/2

cmap = plt.get_cmap("plasma")
colors = [cmap(i) for i in np.linspace(0,1,len(lags))]

os.makedirs(os.path.join(cd,'figures','D_check'),exist_ok=True)

#%% Main
ws=Data_met_sta.ws
wd=Data_met_sta.wd
ti=Data_met_sta.ws_std*100
 
D_T=Data_met_sta['D_res_air_temp_rec']**0.5
space_lag=D_T.lag*ws
D_T=D_T.where(ti<max_ti)


#daily average
hour=np.array([(t-np.datetime64(str(t)[:10]))/np.timedelta64(1,'h') for t in D_T.time.values])

f_avg_all=np.zeros((len(hour_avg),len(D_T.height)))
for i_h in range(len(D_T.height)):
    f=D_T.isel(height=i_h,lag=150).values
    real=~np.isnan(f)
    f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_stat(x,np.nanmean), bins=bin_hour)[0]
 
    f_avg_all[:,i_h]=f_avg

    
T_avg_all=np.zeros((len(hour_avg),len(wd_avg),len(D_T.height)))
for i_h in range(len(D_T.height)):
    f=Data_met_sta.air_temp_rec.isel(height=i_h).values
    real=~np.isnan(f+wd.isel(height=2).values)
    f_avg= stats.binned_statistic_2d(hour[real],+wd.isel(height=2).values[real], f[real],statistic=lambda x:utl.filt_stat(x,np.nanmean), bins=[bin_hour,bin_wd])[0]
    T_avg_all[:,:,i_h]=f_avg
    
T_std_all=np.zeros((len(hour_avg),len(wd_avg),len(D_T.height)))
for i_h in range(len(D_T.height)):
    f=Data_met_sta.air_temp_rec_std.isel(height=i_h).values
    real=~np.isnan(f+wd.isel(height=2).values)
    f_avg= stats.binned_statistic_2d(hour[real],+wd.isel(height=2).values[real], f[real],statistic=lambda x:utl.filt_stat(x,np.nanmean), bins=[bin_hour,bin_wd])[0]
    T_std_all[:,:,i_h]=f_avg
    
ws_std_all=np.zeros((len(hour_avg),len(D_T.height)))
for i_h in range(len(D_T.height)):
    f=Data_met_sta.ws_std.isel(height=i_h).values
    real=~np.isnan(f)
    f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_stat(x,np.nanmean), bins=bin_hour)[0]
 
    ws_std_all[:,i_h]=f_avg


#all high frequency data
for f in files:
    plt.figure(figsize=(18,6))
    data=xr.open_dataset(f)
    
    #structure function
    dt_met=np.median(np.diff(data.time))/np.timedelta64(1,'s')
    
    
    D_res = np.zeros((len(lags),len(data.height)))
    x_res=data['air_temp_rec'].rolling(time=int(config['structure_func_resampling']/dt_met), center=True).mean()
    for i_h in range(len(data.height)):
        ax=plt.subplot(1,len(data.height),i_h+1)
        plt.plot(data.time,x_res.isel(height=i_h),'k')
        i_lag=0
        for lag in lags:
            dsq_res = ((x_res.shift(time=-lag) - x_res)**2).mean(dim='time')
            
            plt.plot(data.time,x_res.shift(time=-lag).isel(height=i_h),color=colors[i_lag],
                     label=f'Lag={lag} s: {np.float64(np.round(dsq_res.isel(height=i_h)**0.5,3))}',alpha=0.5)
            
            i_lag+=1
        plt.xlabel('Time (UTC)')
        plt.title(r'$z='+str(int(data.height[i_h]))+'$ m')
        plt.grid()
        ax.set_xticks(data.time[[0,299,-1]])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        plt.ylim([15,33])
        plt.legend()
        if i_h==0:
            plt.ylabel(r'T [$^\circ$C]')
            
    plt.savefig(os.path.join(cd,'figures','D_check',os.path.basename(f).replace('nc','D.png')))
    plt.close()
        

#%% Plots
t1=103
t2=110
plt.figure()
plt.pcolor(D_T.time[t1*144:t2*144],D_T.lag,D_T.isel(height=0,time=slice(t1*144,t2*144)).T,vmin=0,vmax=1,cmap='hot')
plt.colorbar()

plt.figure()
plt.plot(D_T.time,D_T.isel(lag=150,height=0))
# plt.plot(ws.time,ws.isel(height=0))
plt.grid()



plt.figure(figsize=(18,8))
theta=np.radians(np.append(wd_avg,wd_avg[0]+360))
for i_h in range(len(D_T.height)):
    ax = plt.subplot(2,len(D_T.height),i_h+1,projection='polar')
    plt.contourf(theta,hour_avg,np.vstack([T_avg_all[:,:,i_h].T,T_avg_all[:,0,i_h]]).T,np.arange(5,31,1),extend='both',cmap='hot')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    ax = plt.subplot(2,len(D_T.height),i_h+1+len(D_T.height),projection='polar')
    plt.contourf(theta,hour_avg,np.vstack([T_std_all[:,:,i_h].T,T_std_all[:,0,i_h]]).T,np.arange(0.05,0.251,0.01),extend='both',cmap='plasma')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    
   
    # plt.xlabel('Hour (UTC)')
    # plt.ylabel(r'St.dev. of $T$ [$^\circ$C]')
    # plt.ylim([0.05,0.27])
    # plt.grid()
    # plt.legend()
plt.tight_layout()


plt.figure()
for i_h in range(len(D_T.height)):
    plt.plot(hour_avg,ws_std_all[:,i_h],label=r'$z='+str(int(data.height[i_h]))+'$ m')
plt.xlabel('Hour (UTC)')
plt.ylabel(r'St.dev. of $u$ [m $s^{-1}$]')
plt.grid()
plt.legend()