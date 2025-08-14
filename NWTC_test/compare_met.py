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
source_waked=os.path.join(cd,'data/turbine_wakes.nc')
source_met_sta=os.path.join(cd,'data/nwtc.m5.c1.corr.nc')#source of met stats
sigma_met=0.1#[C] uncertaiinty of met measurements [St Martin et al. 2016]
site_trp= {'ASSIST10':'Site 4.0','ASSIST11':'Site 3.2'}

#user
unit='ASSIST11'#assist id
sel_height=87#[m] height to select wind conditions
var_trp='temperature'#selected variable in TROPoe data
var_met='temperature'#selected variable in M5 data

#stats
p_value=0.05#for CI
max_height=200#[m] maximum height
max_f=40#[C] max threshold of selected variable
min_f=-5#[C] min threshold of selected variable
max_time_diff=10#[s] maximum difference in time between met and TROPoe
perc_lim=[1,99] #[%] percentile filter before feature selection
 
#graphics
cmap = plt.get_cmap("viridis")
zooms=[['2022-05-19','2022-05-21'],
       ['2022-07-23','2022-07-27'],
       ['2022-08-08','2022-08-13']]

rf_vars=['CBH','Richardson number','Wind speed','Wind direction']

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#read and align data
Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.bias.nc'))
Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.a1.{unit}.corr.nc'))

Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner",exclude=["height"])

#read wake data
waked=xr.open_dataset(source_waked)

#read met stats
Data_met_sta=xr.open_dataset(source_met_sta)

#zeroing
importance={}
importance_std={}

#%% Main

#save cbh
cbh=Data_trp.cbh.where(Data_trp.cbh!=np.nanpercentile(Data_trp.cbh,10)).where(Data_trp.cbh!=np.nanpercentile(Data_trp.cbh,90))

#height interpolation
Data_trp=Data_trp.interp(height=Data_met.height)

#QC
Data_trp=Data_trp.where(Data_trp.qc==0)
print(f"{int(np.sum(Data_trp.qc!=0))} points fail QC in TROPoe")

Data_met=Data_met.where(np.abs(Data_met.time_diff)<=max_time_diff)
print(f"{int(np.sum(Data_met.time_diff>max_time_diff))} points fail max_time_diff")

#remove wake
Data_trp['waked']=waked[site_trp[unit]].interp(time=Data_trp.time)
f_trp=Data_trp[var_trp].where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
sigma_trp=Data_trp[f"sigma_{var_trp}"].where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_trp['waked'].sum(dim='turbine')>0))} wake events at {site_trp[unit]} excluded")

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

#feature importance

#preconditioning
Ri=Data_met_sta.Ri_3_122.interp(time=Data_trp.time)
logRi=np.log10(np.abs(Ri.values)+1)*np.sign(Ri.values)

ws=Data_met_sta.ws.sel(height=sel_height).interp(time=Data_trp.time).values

cos_wd=np.cos(np.radians(Data_met_sta.wd.sel(height=sel_height))).interp(time=Data_trp.time).values
sin_wd=np.sin(np.radians(Data_met_sta.wd.sel(height=sel_height))).interp(time=Data_trp.time).values
wd=np.degrees(np.arctan2(sin_wd,cos_wd))%360

X=np.array([utl.perc_filt(cbh.values,perc_lim),
            utl.perc_filt(logRi,perc_lim),
            utl.perc_filt(ws,perc_lim),
            utl.perc_filt(wd,perc_lim)]).T

#importance for signed error
plt.figure(figsize=(18,8))

i_h=0
for h in height:
    y=utl.perc_filt(diff.sel(height=h).values,perc_lim)
    reals=~np.isnan(np.sum(X,axis=1)+y)
    importance[h],importance_std[h],y_pred,test_mae,train_mae,best_params=utl.RF_feature_selector(X[reals,:],y[reals])
    
    for i_x in range(len(rf_vars)):
        ax=plt.subplot(len(height), len(rf_vars),i_h*len(rf_vars)+i_x+1)
        plt.plot(X[:,i_x],y,'.k',alpha=0.05)
        if i_x==0:
            plt.ylabel(r'$\Delta T$ [$^\circ$C]'+'\n ($z='+str(h)+r'$ m)')
        else:
            ax.set_yticklabels([])
        if i_h==len(height)-1:
            plt.xlabel(rf_vars[i_x])
        else:
            ax.set_xticklabels([])
        plt.ylim([-3,3])
        plt.grid()
    i_h+=1

#%% Plots

#time series of T
fig=plt.figure(figsize=(18,10))
for i_h in range(len(height)):
    ax=plt.subplot(len(height),1,i_h+1)
    plt.plot(time,f_met,'-k',alpha=0.25)
    plt.plot(time,f_met.isel(height=i_h),'-k',label='M5')
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
    plt.plot(time,diff.isel(height=i_h),'-k',label='TROPoe-M5')
    plt.plot(time,Data_trp.bias.isel(height=i_h),'-b',label='Prior bias')
    plt.ylim([-3,3])
    plt.grid()
    plt.ylabel(r'$\Delta T$' +'\n (TROPoe-M5)'+r'[$^\circ$C]')
    if i_h==len(height)-1:
        plt.xlabel('Time (UTC)')
    plt.text(time[10],2,r'$z='+str(height[i_h])+r'$ m',bbox={'alpha':0.5,'color':'w'})
plt.legend(draggable=True)

#selected time series
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
        ax=fig.add_subplot(gs[len(height)-i_h,i_z])
        t1=np.datetime64(zoom[0]+'T00:00:00')
        t2=np.datetime64(zoom[1]+'T00:00:00')
        sel=(time>=t1)*(time<=t2)
        precip=Data_met.precip.values[:,0]>0
       
        for t in time[sel*precip]:
            plt.plot([t,t],[-5,35],'b',alpha=0.25)
        plt.plot(time[sel],f_met.values[sel,:],'k',alpha=0.25)
        plt.plot(time[sel][0],0,'k',alpha=0.25,label='M5 (all heights)')
        plt.plot(time[sel],f_met.isel(height=i_h).values[sel],'k',label='M5')
        plt.plot(time[sel],f_trp.isel(height=i_h).values[sel],'.r',markersize=3,label='TROPoe')
        
        ax.set_ylim([-1,35])
        ax.set_xticks(np.arange(t1,t2+np.timedelta64(1,'D'),np.timedelta64(1,'D')))
        if i_z>0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'$T$ [$^\circ$C]')
        if i_h==0:
            plt.xticks(rotation=30)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
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
            else:
                ax.set_yticklabels([])
        
        i_z+=1  
plt.legend(draggable=True)


#linear regression
matplotlib.rcParams['font.size'] = 14
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
                     f_trp.isel(height=i_h).values,ax=ax,cax=cax,bins=50,legend=i_h==0)
    
    ax.set_xlim([0,30])
    ax.set_ylim([0,30])
    ax.set_xticks([0,10,20,30])
    ax.set_yticks([0,10,20,30])
    ax.grid(True)
    ax.set_xlabel(r'$T$ (M5) [$^\circ$C]')
    if i_h==0:
        ax.set_ylabel(r'$T$ (TROPoe) [$^\circ$C]')
        plt.legend(draggable=True)
    else:
        ax.set_yticklabels([])
        
    ax=fig.add_subplot(gs[1,i_h])
    
    plt.hist(diff.isel(height=i_h),bins=bins,color='k',alpha=0.25,density=True)
    plt.plot(bins,norm.pdf(bins,loc=diff.isel(height=i_h).mean(),
                               scale=diff.isel(height=i_h).std()),'k',label='Data')
    plt.plot(bins,norm.pdf(bins,loc=0,scale=(sigma_trp.isel(height=i_h).mean()**2+sigma_met**2)**0.5),'r',label='Theory')
    ax.fill_between(bins,norm.pdf(bins,loc=0,scale=(sigma_trp.isel(height=i_h).min()**2+sigma_met**2)**0.5),
                         norm.pdf(bins,loc=0,scale=(sigma_trp.isel(height=i_h).max()**2+sigma_met**2)**0.5),color='r',alpha=0.25)
    ax.set_yscale('log')
    plt.grid()
    if i_h==0:
        ax.set_ylabel('PDF')
        plt.legend(draggable=True)
    else:
        ax.yaxis.set_major_formatter(NullFormatter())
    
    plt.xlabel(r'$\Delta T$ (TROPoe-M5) [$^\circ$C]')
    plt.ylim([0.01,10])
        
#importance
matplotlib.rcParams['font.size'] = 12
plt.figure(figsize=(14,4))
cmap=matplotlib.cm.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0,1,len(height))]
ctr=0
for h in height:
    plt.bar(np.arange(len(rf_vars))+ctr/len(height)/2,importance[h],color=colors[ctr],width=0.1,yerr=importance_std[h],label=r'$z='+str(h)+r"$ m",capsize=5,linewidth=2)
    ctr+=1

plt.xticks(np.arange(len(rf_vars))+(len(height)-1)/len(height)/4,labels=rf_vars)
plt.grid()
plt.ylabel('Feauture importance')
plt.legend()

