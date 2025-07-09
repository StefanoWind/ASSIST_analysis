# -*- coding: utf-8 -*-
"""
Cluster profiles by atmospheric stability
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
sys.path.append(os.path.join(cd,'../utils'))
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import utils as utl
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs

#dataset
source_stab=os.path.join(cd,'data/nwtc/nwtc.m5.c0/*nc')#source of met stats
source_waked=os.path.join(cd,'data/turbine_wakes.nc')
height_sel=119#[m]
max_height=200#[m]
g=9.81#[m/s^2] gravity acceleration
cp=1005#[J/KgK] #air heat capacity
max_f=40#[C]
min_f=-5#[C]
max_time_diff=10#[s]

#user
unit='ASSIST11'#assist id
var_trp='temperature'
var_met='temperature'#selected temperature variable in M5 data

#stats
stab_classes_uni=['S','NS','N','NU','U']
stab_classes={'S':[0,200],
              'NS':[200,500],
              'N1':[500,np.inf],
              'N2':[-np.inf,-500],
              'NU':[-500,-200],
              'U':[-200,0]}#stability classes from Obukhov length [Hamilton and Debnath, 2019]
p_value=0.05

#graphics
cmap = plt.get_cmap("coolwarm")

#%% Initialization

#read and align data
Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.nc'))
Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.b0.{unit}.nc'))

Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner")

#QC
Data_trp=Data_trp.where(Data_trp.qc==0)
Data_met=Data_met.where(Data_met.time_diff<=max_time_diff)

#read met data
files=glob.glob(source_stab)
met=xr.open_mfdataset(files)
L=met.L.mean(dim="height_kin").interp(time=Data_trp.time)

#hour
hour=[(t-np.datetime64(str(t)[:10]))/np.timedelta64(1,'h') for t in Data_met.time.values]
Data_met['hour']=xr.DataArray(data=hour,coords={'time':Data_met.time.values})

#read wake data
waked=xr.open_dataset(source_waked)

#graphics
colors=[cmap(val) for val in np.linspace(0, 1, len(stab_classes_uni))]

#%% Main

#stability class
stab_class=xr.DataArray(data=['null']*len(L.time),coords={'time':L.time})

for s in stab_classes.keys():
    sel=(L>=stab_classes[s][0])*(L<stab_classes[s][1])
    if s=='N1' or s=='N2':
        s='N'
    stab_class=stab_class.where(~sel,other=s)
    
#remove wakes
Data_trp['waked']=waked['Site 3.2'].interp(time=Data_trp.time)
f_trp=Data_trp[var_trp].where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_trp['waked'].sum(dim='turbine')>0))} wake events at Site 3.2 excluded")

Data_met['waked']=waked['M5'].interp(time=Data_met.time)
f_met=Data_met[var_met].where(Data_met['waked'].sum(dim='turbine')==0).sel(height_therm=slice(0,max_height))
f_met=f_met.rename({'height_therm':'height'})
print(f"{int(np.sum(Data_met['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")

#remove outliers
f_trp=f_trp.where(f_trp>=min_f).where(f_trp<=max_f)
f_met=f_met.where(f_met>=min_f).where(f_met<=max_f)

#remove common nans
real=np.isnan(f_trp).sum(dim="height")+np.isnan(f_met).sum(dim="height")==0
f_trp=f_trp[real]
f_met=f_met[real]

# #stats
# f_trp_avg=np.zeros((len(f_trp.height),len(stab_classes_uni)))
# f_trp_low=np.zeros((len(f_trp.height),len(stab_classes_uni)))
# f_trp_top=np.zeros((len(f_trp.height),len(stab_classes_uni)))
# for i_sc in range(len(stab_classes_uni)):
#     sc=stab_classes_uni[i_sc]
#     for i_h in range(len(f_trp.height)):
#         f_sel=f_trp.isel(height=i_h).where(stab_class==sc).values
#         f_trp_avg[i_h,i_sc]=utl.filt_stat(f_sel,np.nanmedian)
#         f_trp_low[i_h,i_sc]=utl.filt_BS_stat(f_sel,np.nanmedian,p_value/2*100)
#         f_trp_top[i_h,i_sc]=utl.filt_BS_stat(f_sel,np.nanmedian,(1-p_value/2)*100)

# trp_stats=xr.Dataset()
# trp_stats['f_avg']=xr.DataArray(data=f_trp_avg,coords={'height':f_trp.height,'_class':stab_classes_uni})
# trp_stats['f_low']=xr.DataArray(data=f_trp_low,coords={'height':f_trp.height,'_class':stab_classes_uni})
# trp_stats['f_top']=xr.DataArray(data=f_trp_top,coords={'height':f_trp.height,'_class':stab_classes_uni})

# f_met_avg=np.zeros((len(f_met.height),len(stab_classes_uni)))
# f_met_low=np.zeros((len(f_met.height),len(stab_classes_uni)))
# f_met_top=np.zeros((len(f_met.height),len(stab_classes_uni)))
# for i_sc in range(len(stab_classes_uni)):
#     sc=stab_classes_uni[i_sc]
#     for i_h in range(len(f_met.height)):
#         f_sel=f_met.isel(height=i_h).where(stab_class==sc).values
#         f_met_avg[i_h,i_sc]=utl.filt_stat(f_sel,np.nanmean)
#         f_met_low[i_h,i_sc]=utl.filt_BS_stat(f_sel,np.nanmean,p_value/2*100)
#         f_met_top[i_h,i_sc]=utl.filt_BS_stat(f_sel,np.nanmean,(1-p_value/2)*100)

# met_stats=xr.Dataset()
# met_stats['f_avg']=xr.DataArray(data=f_met_avg,coords={'height':f_met.height,'_class':stab_classes_uni})
# met_stats['f_low']=xr.DataArray(data=f_met_low,coords={'height':f_met.height,'_class':stab_classes_uni})
# met_stats['f_top']=xr.DataArray(data=f_met_top,coords={'height':f_met.height,'_class':stab_classes_uni})


#stats
bins_hour=np.arange(0,25,4)
hour_avg=(bins_hour[:-1]+bins_hour[1:])/2
f_trp_avg=np.zeros((len(f_trp.height),len(hour_avg)))
f_trp_low=np.zeros((len(f_trp.height),len(hour_avg)))
f_trp_top=np.zeros((len(f_trp.height),len(hour_avg)))
for i_h in range(len(bins_hour)-1):
    h1=bins_hour[i_h]
    h2=bins_hour[i_h+1]
    for i_z in range(len(f_trp.height)):
        f_sel=f_trp.isel(height=i_z).where(Data_met.hour>=h1).where(Data_met.hour<h2).values
        f_trp_avg[i_z,i_h]=utl.filt_stat(f_sel,np.nanmedian)
        f_trp_low[i_z,i_h]=utl.filt_BS_stat(f_sel,np.nanmedian,p_value/2*100)
        f_trp_top[i_z,i_h]=utl.filt_BS_stat(f_sel,np.nanmedian,(1-p_value/2)*100)

trp_stats=xr.Dataset()
trp_stats['f_avg']=xr.DataArray(data=f_trp_avg,coords={'height':f_trp.height,'hour':hour_avg})
trp_stats['f_low']=xr.DataArray(data=f_trp_low,coords={'height':f_trp.height,'hour':hour_avg})
trp_stats['f_top']=xr.DataArray(data=f_trp_top,coords={'height':f_trp.height,'hour':hour_avg})

f_met_avg=np.zeros((len(f_met.height),len(hour_avg)))
f_met_low=np.zeros((len(f_met.height),len(hour_avg)))
f_met_top=np.zeros((len(f_met.height),len(hour_avg)))
for i_h in range(len(bins_hour)-1):
    h1=bins_hour[i_h]
    h2=bins_hour[i_h+1]
    for i_z in range(len(f_met.height)):
        f_sel=f_met.isel(height=i_z).where(Data_met.hour>=h1).where(Data_met.hour<h2).values
        f_met_avg[i_z,i_h]=utl.filt_stat(f_sel,np.nanmedian)
        f_met_low[i_z,i_h]=utl.filt_BS_stat(f_sel,np.nanmedian,p_value/2*100)
        f_met_top[i_z,i_h]=utl.filt_BS_stat(f_sel,np.nanmedian,(1-p_value/2)*100)

met_stats=xr.Dataset()
met_stats['f_avg']=xr.DataArray(data=f_met_avg,coords={'height':f_met.height,'hour':hour_avg})
met_stats['f_low']=xr.DataArray(data=f_met_low,coords={'height':f_met.height,'hour':hour_avg})
met_stats['f_top']=xr.DataArray(data=f_met_top,coords={'height':f_met.height,'hour':hour_avg})

#%% Plots

#average profiles
plt.figure(figsize=(18,4))

for i_sc in range(len(stab_classes_uni)):
    plt.subplot(1,len(stab_classes_uni),i_sc+1)
    plt.plot(met_stats.f_avg.isel(_class=i_sc),met_stats.height,'.-k',label='Met')
    plt.fill_betweenx(met_stats.height,met_stats.f_low.isel(_class=i_sc),
                                      met_stats.f_top.isel(_class=i_sc),
                                      color='k',alpha=0.25)
    plt.plot(trp_stats.f_avg.isel(_class=i_sc),trp_stats.height,'.-r',label='TROPoe')
    plt.fill_betweenx(trp_stats.height,trp_stats.f_low.isel(_class=i_sc),
                                      trp_stats.f_top.isel(_class=i_sc),
                                      color='r',alpha=0.25)
    plt.plot(-g/cp*f_trp.height+met_stats.f_avg.isel(_class=i_sc).isel(height=0),trp_stats.height,'--k')
    
    plt.xlim([15,25])
    plt.grid()
    plt.xlabel(r'$T$ [$^\circ$C]')
    if i_sc==0:
        plt.ylabel(r'$z$ [m]')
    
    plt.title(stab_classes_uni[i_sc])
plt.tight_layout()
plt.legend()
    

#average profiles
plt.figure(figsize=(18,4))

for i_h in range(len(hour_avg)):
    plt.subplot(1,len(hour_avg),i_h+1)
    plt.plot(met_stats.f_avg.isel(hour=i_h),met_stats.height,'.-k',label='Met')
    plt.fill_betweenx(met_stats.height,met_stats.f_low.isel(hour=i_h),
                                      met_stats.f_top.isel(hour=i_h),
                                      color='k',alpha=0.25)
    plt.plot(trp_stats.f_avg.isel(hour=i_h),trp_stats.height,'.-r',label='TROPoe')
    plt.fill_betweenx(trp_stats.height,trp_stats.f_low.isel(hour=i_h),
                                      trp_stats.f_top.isel(hour=i_h),
                                      color='r',alpha=0.25)
    plt.plot(-g/cp*f_trp.height+met_stats.f_avg.isel(hour=i_h).isel(height=0),trp_stats.height,'--k')
    
    plt.xlim([15,30])
    plt.grid()
    plt.xlabel(r'$T$ [$^\circ$C]')
    if i_sc==0:
        plt.ylabel(r'$z$ [m]')
    
    plt.title(hour_avg[i_h])
plt.tight_layout()
plt.legend()
    
    