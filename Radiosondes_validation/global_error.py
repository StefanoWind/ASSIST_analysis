# -*- coding: utf-8 -*-
"""
Global error stats
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import warnings
import matplotlib
import pandas as pd
import glob 
from scipy.stats import binned_statistic

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 20

#%% Inputs
source_trp='C:/Users/SLETIZIA/OneDrive - NREL/Desktop/Main/AWAKEN/ASSIST_analysis/ASSIST_analysis/Met_validation/data/assist-10/*nc'
source_rsn='data/sgpsondewnpnS6.b1/*cdf'

#dataset
z0_sonde=0#[m] initial height of radiosondes
unc_rsn=0.15#[C] 1-sigma uncertainty of radiosondes [https://www.arm.gov/publications/tech_reports/handbooks/sonde_handbook.pdf]
window=1#smoothing window

#qc
max_gamma=1
max_rmsa=5
min_lwp=5#[g/m^1]

#graphics
max_height=2000#[m]

#%% Initialization
files_rsn=glob.glob(source_rsn)
files_trp=glob.glob(source_trp)

height=xr.open_dataset(files_trp[0]).height.values*1000
height=height[height<=max_height]

#zeroing
T_rsn=[]
T_std_rsn=[]
T_trp=[]
sigmaT_trp=[]
time_rsn=np.array([],dtype='datetime64')
time_trp=np.array([],dtype='datetime64')

#%% Main
for f in files_rsn:
    #load radiosondes
    Data_rsn=xr.open_dataset(f)
    
    #extract profiles
    time=Data_rsn['time'].values
    asc=Data_rsn['asc'].values
    T=Data_rsn['tdry'].rolling(time=window,center=True).mean().values
    
    tnum=np.float64(time)/10**9
    height_rsn=cumtrapz(asc,tnum,initial=z0_sonde)
    
    #time matching
    date=utl.datestr(np.nanmedian(tnum[height_rsn<=max_height]),'%Y%m%d')
    match=result = [s for s in files_trp if date in s]
    mean_time=np.datetime64('1970-01-01T00:00:00')+np.timedelta64(1, 's')*np.nanmedian(tnum[height_rsn<=max_height])
    
    if len(match)==1:
        
        #load tropoe
        Data_trp=xr.open_dataset(match[0])
        
        #qc
        Data_trp['cbh'][Data_trp['lwp']<min_lwp]=Data_trp['height'].max()#remove clouds with low lwp
        Data_trp['temperature_qc']=Data_trp['temperature'].where(Data_trp['gamma']<=max_gamma)\
            .where(Data_trp['rmsa']<=max_rmsa).where(Data_trp['height']<=Data_trp['cbh'])#filter temperature
        
        #interpolation
        Data_sel=Data_trp.interp(time=mean_time,height=height/1000)
        
        #stack data
        if np.sum(~np.isnan(Data_sel.temperature)>0):
            T_trp=utl.vstack(T_trp,Data_sel.temperature.values)
            sigmaT_trp=utl.vstack(sigmaT_trp,Data_sel.sigma_temperature.values)
            
            T_rsn=utl.vstack(T_rsn,np.interp(height,height_rsn,T))
            time_rsn=np.append(time_rsn,mean_time)
    print(f)

unc=(np.nanmean(sigmaT_trp,axis=0)**2+unc_rsn**2)**0.5

#%% Plots
plt.close('all')
fig=plt.figure(figsize=(25,10))
gs = gridspec.GridSpec(2, 4,height_ratios=[1,10],width_ratios=[3,5,5,5])
ax=fig.add_subplot(gs[1,0])
plt.plot(np.nanmean(T_trp-T_rsn,axis=0),height,'.-k',label='Mean')
plt.plot(np.nanstd(T_trp-T_rsn,axis=0),height,'.-r',label='StDev')
ax.fill_betweenx(height,-unc,unc,color='b',alpha=0.25)
plt.ylim([0,height[-1]])
plt.grid()
plt.xlabel(r'$\Delta T$ (sondes-TROPoe) [$^\circ$C]')
plt.ylabel('Height [m]')
plt.legend()

ax=fig.add_subplot(gs[1,1])
cf=plt.contourf(np.arange(len(time_rsn)),height,T_rsn.T,np.arange(15,37,0.5),cmap='hot',extend='both')
plt.contour(np.arange(len(time_rsn)),height,T_rsn.T,np.arange(15,37,0.5),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,height[-1]])

cax=fig.add_subplot(gs[0,1])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(15,36,5))
cb.set_label(label='Temperature (sondes) [$^\circ$C]', labelpad=-100)

ax=fig.add_subplot(gs[1,2])
cf=plt.contourf(np.arange(len(time_rsn)),height,T_trp.T,np.arange(15,37,0.5),cmap='hot',extend='both')
plt.contour(np.arange(len(time_rsn)),height,T_trp.T,np.arange(15,37,0.5),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,height[-1]])

cax=fig.add_subplot(gs[0,2])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(15,36,5))
cb.set_label(label='Temperature (TROPoe) [$^\circ$C]', labelpad=-100)

ax=fig.add_subplot(gs[1,3])
cf=plt.contourf(np.arange(len(time_rsn)),height,T_trp.T-T_rsn.T,np.arange(-5,5.1,0.25),cmap='seismic',extend='both')
plt.contour(np.arange(len(time_rsn)),height,T_trp.T-T_rsn.T,np.arange(-5,5.1,0.25),colors='k',alpha=0.25,linewidths=1)
plt.xlabel('Sonde launch #')
ax.set_yticklabels([])
plt.ylim([0,height[-1]])

cax=fig.add_subplot(gs[0,3])
cb=plt.colorbar(cf,cax,orientation='horizontal',ticks=np.arange(-5,5.1,2.5))
cb.set_label(label='$\Delta T$ (sondes-TROPoe) [$^\circ$C]', labelpad=-100)