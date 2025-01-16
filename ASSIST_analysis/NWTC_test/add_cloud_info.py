# -*- coding: utf-8 -*-
"""
Add cloud information to spectra
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import matplotlib.gridspec as gridspec
import xarray as xr
import glob
import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_irs=os.path.join(cd,'data','*.irs.nc')
source_cbh=os.path.join(cd,'data','*.cbh.nc')
cloud_window=np.timedelta64(3600,'s')#cloud search window
wnum_cbh=900#[cm^-1] wnum sensitive to clouds
time_res= np.timedelta64(60,'s')#time resolution when uniformily sampling the cbh
min_da=0.5#minimum data availability in a window

#graphics
N_days_plot=7#number of days to plot in one figure

#%% Initialization
files_irs=np.array(sorted(glob.glob(source_irs)))
Data_irs=xr.open_mfdataset(files_irs)
files_cbh=np.array(sorted(glob.glob(source_cbh)))
Data_cbh=xr.open_mfdataset(files_cbh)

#drop duplicates
_,time_irs_uni=np.unique(Data_irs.time,return_index =True)
Data_irs=Data_irs.isel(time=time_irs_uni)
_,time_cbh_uni=np.unique(Data_cbh.time,return_index =True)
Data_cbh=Data_cbh.isel(time=time_cbh_uni)

#%% Main

#time operations
time_int=np.arange(Data_irs.time.values[0],Data_irs.time.values[-1]+time_res,time_res)
window=int(cloud_window/time_res)

#build cloud flag
cbh_int=Data_cbh['cbh'].interp(time=time_int).fillna(0)
cbh_rol=cbh_int.rolling(time=window,min_periods=int(min_da*window),center=True).max()
Data_irs['cloud_flag']=cbh_rol.interp(time=Data_irs.time)>0

#radiance std at selected wnum
rad_sel=Data_irs.rad.sel(wnum=wnum_cbh,method='nearest')
rad_int=rad_sel.interp(time=time_int).mean(dim='channel')
rad_std=rad_int.rolling(time=window,min_periods=int(min_da*window),center=True).std()
Data_irs['rad_std_'+str(wnum_cbh)]=rad_std.interp(time=Data_irs.time)

#%% Output
Data_irs.to_netcdf(os.path.join(cd,'data',
                 str(np.min(Data_irs.time.values))[:10].replace('-','')\
            +'.'+str(np.max(Data_irs.time.values))[:10].replace('-','')+'.irs.nc').replace('irs','irs_with_cbh'))

#%% Plots
os.makedirs(os.path.join(cd,'figures','cbh'),exist_ok=True)

time_bins=np.arange(Data_irs.time.values[0],Data_irs.time.values[-1]+np.timedelta64(N_days_plot, 'D'),np.timedelta64(N_days_plot, 'D'))
for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    Data_irs_sel=Data_irs.sel(time=slice(t1,t2))
    Data_cbh_sel=Data_cbh.sel(time=slice(t1,t2))
    fig=plt.figure(figsize=(18,8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.5,10,5])
    ax=fig.add_subplot(gs[0,0])
    for i in range(5):
        plt.plot(Data_irs_sel.time,Data_irs_sel.cloud_flag.where(Data_irs_sel.cloud_flag==1)+i,'.k',markersize=2)
    plt.yticks([])
    plt.xlim([-0.1,1.1])
    plt.grid()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.xlim([t1,t2])
    plt.title('Cloud flag (window='+str(cloud_window.astype('int'))+' s)')
    
    ax=fig.add_subplot(gs[1,0])
    plt.plot(Data_cbh_sel.time,Data_cbh_sel.cbh,'.k',alpha=0.5,label='Original',markersize=3)
    plt.ylabel('First CBH [m]')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.xlim([t1,t2])
    plt.ylim([0,cbh_int.max()])
    plt.grid()
    
    ax=fig.add_subplot(gs[2,0])
    plt.semilogy(Data_irs_sel.time,Data_irs_sel['rad_std_'+str(wnum_cbh)],'.k',markersize=2)
    plt.ylabel(r'$\sigma (B)$ at $\tilde{\nu}='+str(wnum_cbh)+ '$ cm $^{-1}$')
    plt.xlabel('Time (UTC)')
    plt.xlim([t1,t2])
    plt.ylim([0,rad_std.max()])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.grid()
    
    plt.savefig(os.path.join(cd,'figures','cbh',str(t1)[:10]+'.'+str(t2)[:10]+'.png'))
    plt.close()