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
import glob
import yaml
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_met_sta=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats
source_config=os.path.join(cd,'configs','config.yaml')
source_waked=os.path.join(cd,'data/turbine_wakes.nc')

#site
unit='ASSIST11'
var_trp='temperature'#selected variable in TROPoe data
var_met='temperature'#selected variable in M5 data

#stats
p_value=0.05#for CI
max_height=200#[m]
max_f=40#[C] max threshold of selected variable
min_f=-5#[C] min threshold of selected variable
max_time_diff=10#[s] maximum difference in time between met and TROPoe
perc_lim=[1,99] #[%] percentile fitler before feature selection
rf_vars=['CBH','Ri','Wind speed','Wind direction']
sel_height=87


#graphics
cmap = plt.get_cmap("plasma")

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#read wake data
waked=xr.open_dataset(source_waked)

#read met stats
files=glob.glob(source_met_sta)
Data_met_sta=xr.open_mfdataset(files)


#zeroing
importance={}
importance_std={}


#%% Main


#read and align data
Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.bias.nc'))
Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.a1.{unit}.nc'))
Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner",exclude=["height"])

#save cbh
cbh=Data_trp.cbh.where(Data_trp.cbh!=np.nanpercentile(Data_trp.cbh,10)).where(Data_trp.cbh!=np.nanpercentile(Data_trp.cbh,90))

#height interpolation
Data_trp=Data_trp.interp(height=Data_met.height)

#QC
Data_trp=Data_trp.where(Data_trp.qc==0)
print(f"{int(np.sum(Data_trp.qc!=0))} points fail QC in TROPoe")

Data_met=Data_met.where(Data_met.time_diff<=max_time_diff)
print(f"{int(np.sum(Data_met.time_diff>max_time_diff))} points fail max_time_diff")

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
    
#met stats synch
ws=Data_met_sta.ws.interp(time=Data_trp.time)

cos_wd=np.cos(np.radians(Data_met_sta.wd)).interp(time=Data_trp.time)
sin_wd=np.sin(np.radians(Data_met_sta.wd)).interp(time=Data_trp.time)
wd=np.degrees(np.arctan2(sin_wd,cos_wd))%360

Ri=Data_met_sta.Ri_3_122.interp(time=Data_trp.time)

bin_Ri=np.nanpercentile(Ri.values.ravel(),np.linspace(1,99,10))
f_avg=np.zeros((len(height),len(bin_Ri)-1))
for i_h in range(len(height)):
    i_Ri=0
    for Ri1,Ri2 in zip(bin_Ri[:-1],bin_Ri[1:]):
        sel_Ri=(Ri>=Ri1)*(Ri<Ri2)
      
        f_sel=diff.isel(height=i_h).where(sel_Ri).values
        f_avg[i_h,i_Ri]=utl.filt_stat(f_sel, np.nanmean)
        i_Ri+=1
        print(i_Ri)
Ri_avg=(bin_Ri[1:]+bin_Ri[:-1])/2
# bias=xr.DataArray(f_avg,coords={'height':height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})

bias=np.zeros(np.shape(diff))
for i_h in range(len(height)):
    bias[:,i_h]=np.interp(Ri,Ri_avg,f_avg[i_h,:])

bias=xr.DataArray(bias,coords=diff.coords)
diff_unb=diff-bias



#feature importance
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
    y=np.abs(utl.perc_filt(diff_unb.sel(height=h).values,perc_lim))
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
plt.figure(figsize=(18,8))
for i_h in range(len(height)):
    ax=plt.subplot(2,len(height),i_h+1)
    plt.plot(Ri,diff.isel(height=i_h),'.k',alpha=0.1)
    plt.plot(Ri_avg,f_avg[i_h,:],'.-r')
    plt.title(np.round(np.nanstd(diff.isel(height=i_h)),3))
    ax.set_xscale('symlog')
    
    ax=plt.subplot(2,len(height),i_h+1+len(height))
    plt.plot(Ri,diff_unb.isel(height=i_h),'.k',alpha=0.1)
    plt.title(np.round(np.nanstd(diff_unb.isel(height=i_h)),3))
    ax.set_xscale('symlog')

    
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


  