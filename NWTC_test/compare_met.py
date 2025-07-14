# -*- coding: utf-8 -*-
"""
Compare tropoe retrievals to met tower data
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
from matplotlib.ticker import NullFormatter,ScalarFormatter
import matplotlib.gridspec as gridspec
import glob
import matplotlib.dates as mdates
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')
source_waked=os.path.join(cd,'data/turbine_wakes.nc')
source_met_sta=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats

#user
unit='ASSIST11'#assist id

#user
var_trp='temperature'
var_met='temperature'#selected temperature variable in M5 data

#stats
p_value=0.05#for CI
max_height=200#[km]
bins_hour=np.arange(25)#[h] hour bins
max_f=40#[C]
min_f=-5#[C]
max_time_diff=10#[s]
 
#graphics
cmap = plt.get_cmap("viridis")
zooms=[['2022-05-19','2022-05-21'],
       ['2022-07-23','2022-07-27'],
       ['2022-08-08','2022-08-13']]
max_cbh=2000

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#read and align data
Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.nc'))
Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.a1.{unit}.nc'))

Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner",exclude=["height"])

waked=xr.open_dataset(source_waked)

files=glob.glob(source_met_sta)
Data_met_sta=xr.open_mfdataset(files)

#%% Main

#interpolation
cbh=Data_trp.cbh.where(Data_trp.cbh!=np.nanpercentile(Data_trp.cbh,10))
Data_trp=Data_trp.interp(height=Data_met.height)

#QC
Data_trp=Data_trp.where(Data_trp.qc==0)
Data_met=Data_met.where(Data_met.time_diff<=max_time_diff)

#remove wake
Data_trp['waked']=waked['Site 3.2'].interp(time=Data_trp.time)
f_trp=Data_trp[var_trp].where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
sigma_trp=Data_trp[f"sigma_{var_trp}"].where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_trp['waked'].sum(dim='turbine')>0))} wake events at Site 3.2 excluded")

Data_met['waked']=waked['M5'].interp(time=Data_met.time)
f_met=Data_met[var_met].where(Data_met['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_met['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")

#remove outliers
f_trp=f_trp.where(f_trp>=min_f).where(f_trp<=max_f)
f_met=f_met.where(f_met>=min_f).where(f_met<=max_f)
    
#extract coords
height=Data_met.height.values
time=Data_met.time.values

#T difference
diff=f_trp-f_met

#hourly stats
tnum=np.float64(time)/10**9
hour=(tnum-np.floor(tnum/(3600*24))*3600*24)/3600

#feature importance

#preconditioning
raise BaseException()
Ri=Data_met_sta.Ri_3_122.interp(time=Data_trp.time)
logRi=np.log10(np.abs(Ri.values)+1)*np.sign(Ri.values)

ws=Data_met_sta.ws.sel(height=87).interp(time=Data_trp.time)

wd=Data_met_sta.wd.sel(height=87).interp(time=Data_trp.time)

ti=Data_met_sta.ws.sel(height=87).interp(time=Data_trp.time)

X=np.array([cbh.values,logRi,ws,wd]).T

importance_sig={}
importance_sig_std={}
for h in height:
    y=diff.sel(height=h).values
    reals=~np.isnan(np.sum(X,axis=1)+y)
    importance_sig[h],importance_sig_std[h],y_pred,test_mae,train_mae,best_params=utl.RF_feature_selector(X[reals,:],y[reals])
    
importance_abs={}
importance_abs_std={}
for h in height:
    y=np.abs(diff.sel(height=h).values)
    reals=~np.isnan(np.sum(X,axis=1)+y)
    importance_abs[h],importance_abs_std[h],y_pred,test_mae,train_mae,best_params=utl.RF_feature_selector(X[reals,:],y[reals])

#%% Plots

plt.close('all')

#time series of T
fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,f_met,'-k',alpha=0.25)
    plt.plot(time,f_met.isel(height=i_h),'-k',label='Met')
    plt.plot(time,f_trp.isel(height=i_h),'-r',label='TROPoe')
    plt.ylim([-5,35])
    plt.grid()
    plt.ylabel(r'$T$ [$^\circ$C]')
    if i_h==len(height)-1:
        plt.xlabel('Time (UTC)')
    plt.text(time[10],25,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
plt.legend()

#time series of DT
fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,diff.isel(height=i_h),'-k',markersize=3,label='TROPoe-met')
    # plt.plot(time,Data['trp_temperature_bias'].isel(height=i_h),'r',label='Prior bias')
    plt.ylim([-3,3])
    plt.grid()
    plt.ylabel(r'$\Delta T$' +'\n (TROPoe-met)'+r'[$^\circ$C]')
    if i_h==len(height)-1:
        plt.xlabel('Time (UTC)')
    plt.text(time[10],2,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
plt.legend()

fig=plt.figure(figsize=(18,10))

durations=[]
for zoom in zooms:
    t1=np.datetime64(zoom[0]+'T00:00:00')
    t2=np.datetime64(zoom[1]+'T00:00:00')
    durations.append(np.float64(t2-t1))
               
gs = gridspec.GridSpec(len(height)+1,len(zooms),width_ratios=durations/durations[0]) 
    
for i_h in range(len(height)):
    i_z=0
    for zoom in zooms:
        ax=fig.add_subplot(gs[i_h+1,i_z])
        t1=np.datetime64(zoom[0]+'T00:00:00')
        t2=np.datetime64(zoom[1]+'T00:00:00')
        sel=(time>=t1)*(time<=t2)
        precip=Data_met.precip.values[:,0]>0
       
        for t in time[sel*precip]:
            plt.plot([t,t],[-5,35],'b',alpha=0.25)
        plt.plot(time[sel],f_met.values[sel,:],'k',alpha=0.25)
        plt.plot(time[sel],f_met.isel(height=i_h).values[sel],'k')
        plt.plot(time[sel],f_trp.isel(height=i_h).values[sel],'.r',markersize=3)
        
        ax.set_ylim([-1,35])
        ax.set_xticks(np.arange(t1,t2+np.timedelta64(1,'D'),np.timedelta64(1,'D')))
        if i_z>0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$T$ [$^\circ$C]')
        if i_h==len(height)-1:
            plt.xticks(rotation=30)
        else:
            ax.set_xticklabels([])
        
        plt.grid()
        
        if i_h==0:
            ax=fig.add_subplot(gs[0,i_z])
            plt.plot(time[sel],cbh.values[sel],'.w')
            plt.ylim([0,15])
            
            ax.set_facecolor('k')
            plt.grid()
            ax.set_xticks(np.arange(t1,t2+np.timedelta64(1,'D'),np.timedelta64(1,'D')))
            ax.set_xticklabels([])
        if i_z==0:
            ax.set_ylabel('CBH [km]')
        
        i_z+=1  

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 

#linear regression
bins=np.arange(-5,5.1,0.05)
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2,len(height)+1,width_ratios=[1]*len(height)+[0.05]) 
for i_h in range(len(height)):
    ax=fig.add_subplot(gs[0,i_h])
    if i_h==len(height)-1:
        cax=fig.add_subplot(gs[0,i_h+1])
    else:
        cax=None
    utl.plot_lin_fit(f_met.isel(height=i_h).values,
                     f_trp.isel(height=i_h).values,ax=ax,cax=cax,bins=50)
    
    ax.set_xlim([0,30])
    ax.set_ylim([0,30])
    ax.set_xticks([0,10,20,30])
    ax.set_yticks([0,10,20,30])
    ax.grid(True)
    ax.set_xlabel(r'$T$ (met) [$^\circ$C]')
    if i_h==0:
        ax.set_ylabel(r'$T$ (TROPoe) [$^\circ$C]')
    else:
        ax.set_yticklabels([])
        
    ax=fig.add_subplot(gs[1,i_h])
    
    plt.hist(diff.isel(height=i_h),bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=diff.isel(height=i_h).mean(),
                               scale=diff.isel(height=i_h).std()),'k',label='Data')
    plt.plot(bins,norm.pdf(bins,loc=0,scale=sigma_trp.isel(height=i_h).mean()),'r',label='Theory')
    ax.fill_between(bins,norm.pdf(bins,loc=0,scale=sigma_trp.isel(height=i_h).max()),
                         norm.pdf(bins,loc=0,scale=sigma_trp.isel(height=i_h).min()),color='r',alpha=0.25)
    ax.set_yscale('log')
    plt.grid()
    if i_h==0:
        ax.set_ylabel('Count')
    else:
        ax.yaxis.set_major_formatter(NullFormatter())
    
    plt.xlabel(r'$\Delta T$ (TROPoe-met) [$^\circ$C]')
    plt.ylim([0.01,10])
plt.legend(draggable=True)
        
    


