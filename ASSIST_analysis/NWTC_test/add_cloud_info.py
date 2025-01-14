# -*- coding: utf-8 -*-
"""
Add cluod information to spectra
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import matplotlib.gridspec as gridspec
import xarray as xr
import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_irs=os.path.join(cd,'data','20220515.20220801.irs.nc')
source_cbh=os.path.join(cd,'data','20220515.20220801.cbh.nc')
cloud_window=np.timedelta64(3600,'s')#cloud search window
wnum_cbh=900#[cm^-1] wnum sensitive to clouds

#graphics
N_days_plot=7

#%% Initialization
Data_irs=xr.open_dataset(source_irs)
Data_cbh=xr.open_dataset(source_cbh)

#%% Main

#build cloud flag
Data_irs['cbh']=Data_cbh['cbh'].interp(time=Data_irs.time)
cloud_flag=np.zeros(len(Data_irs.time))
rad_std=np.zeros((len(Data_irs.time),len(Data_irs.channel)))
for it in range(len(Data_irs.time)):
    t=Data_irs.time.values[it]
    if Data_irs.cbh.sel(time=slice(t-cloud_window/2,t+cloud_window/2)).max()>0:
        cloud_flag[it]=1
    rad_std[it,:]=Data_irs.rad.sel(wnum=wnum_cbh,method='nearest').sel(time=slice(t-cloud_window/2,t+cloud_window/2)).std(dim='time')
    print(it/len(Data_irs.time))
    
Data_irs['cloud_flag']=xr.DataArray(data=cloud_flag,coords={'time':Data_irs.time})
Data_irs['rad_std']=xr.DataArray(data=rad_std,coords={'time':Data_irs.time,'channel':Data_irs.channel})

#%% Output
Data_irs.to_netcdf(source_irs.replace('irs','irs.cbh'))

#%% Plots
os.makedirs(os.path.join(cd,'figures','cbh'),exist_ok=True)

time_bins=np.arange(Data_irs.time.values[0],Data_irs.time.values[-1]+np.timedelta64(N_days_plot, 'D'),np.timedelta64(N_days_plot, 'D'))
for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    Data_irs_sel=Data_irs.sel(time=slice(t1,t2))
    Data_cbh_sel=Data_cbh.sel(time=slice(t1,t2))
    fig=plt.figure(figsize=(18,8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1,10,5])
    ax=fig.add_subplot(gs[0,0])
    plt.plot(Data_irs_sel.time,Data_irs_sel.cloud_flag.where(Data_irs_sel.cloud_flag==1),'.k',markersize=2)
    plt.yticks([])
    plt.xlim([-0.1,0.1])
    plt.grid()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.xlim([t1,t2])
    plt.title('Cloud flag (window='+str(cloud_window.astype('int'))+' s)')
    
    ax=fig.add_subplot(gs[1,0])
    plt.plot(Data_cbh_sel.time,Data_cbh_sel.cbh,'.k',alpha=0.5,label='Original',markersize=3)
    plt.plot(Data_irs_sel.time,Data_irs_sel.cbh,'.g',alpha=0.5,label='Interpolated',markersize=2)
    plt.ylabel('First CBH [m]')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.xlim([t1,t2])
    plt.grid()
    plt.legend()
    
    ax=fig.add_subplot(gs[2,0])
    for channel in Data_irs_sel.channel:
        plt.semilogy(Data_irs_sel.time,Data_irs_sel.rad_std.sel(channel=channel),'.',label=str(channel.values),markersize=2)
    plt.ylabel(r'$\sigma (B)$ at $\tilde{\nu}='+str(wnum_cbh)+ '$ cm $^{-1}$')
    plt.xlabel('Time (UTC)')
    plt.xlim([t1,t2])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.grid()
    plt.legend()
    
    plt.savefig(os.path.join(cd,'figures','cbh',str(t1)[:10]+'.'+str(t2)[:10]+'.png'))
    plt.close()