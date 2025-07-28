# -*- coding: utf-8 -*-
"""
Cluster profiles by atmospheric stability
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(cd,'../utils'))
import xarray as xr
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
import utils as utl
import glob
import warnings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs

#dataset
source_stab=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats
source_waked=os.path.join(cd,'data/turbine_wakes.nc')
source_m2=os.path.join(cd,'data','nwtc.m2.b0.csv')
height_m2=[2,50,80]
time_offset_m2=7#[h]

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
bin_Ri=np.array([-100,-0.25,-0.03,0.03,0.25,100])#bins in Ri [mix of Hamilton 2019 and Aitken 2014]
p_value=0.05
perc_lim=[5,95]

#graphics
cmap = plt.get_cmap("coolwarm")
stab_names={'S':4,'NS':3,'N':2,'NU':1,'U':0}

#%% Initialization

#read and align data
Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.bias.nc'))
Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.a1.{unit}.nc'))

Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner",exclude=["height"])

#read M2 data
Data_m2_df=pd.read_csv(source_m2, parse_dates=[["DATE (MM/DD/YYYY)", "MST"]])
time_m2=np.array([np.datetime64(t) for t in pd.to_datetime(Data_m2_df['DATE (MM/DD/YYYY)_MST'])])+np.timedelta64(time_offset_m2, 'h')
Data_m2=xr.Dataset()
Data_m2['temperature']=xr.DataArray(Data_m2_df.iloc[:,1:].values,coords={'time':time_m2,'height':height_m2})

#interpolate m2 data into common time
tnum_trp=(Data_trp.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
tnum_m2=(Data_m2.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
time_diff_m2=tnum_m2.interp(time=Data_trp.time,method='nearest')-tnum_trp

Data_m2=Data_m2.interp(time=Data_trp.time.values)
Data_m2['time_diff']=time_diff_m2

#QC
Data_trp=Data_trp.where(Data_trp.qc==0)
Data_met=Data_met.where(Data_met.time_diff<=max_time_diff)

#read met data
files=glob.glob(source_stab)
met=xr.open_mfdataset(files)

#read wake data
waked=xr.open_dataset(source_waked)

#%% Main

#QC
Data_trp=Data_trp.where(Data_trp.qc==0)
print(f"{int(np.sum(Data_trp.qc!=0))} points fail QC in TROPoe")

Data_met=Data_met.where(Data_met.time_diff<=max_time_diff)
print(f"{int(np.sum(Data_met.time_diff>max_time_diff))} points fail max_time_diff")

Data_m2=Data_m2.where(Data_m2.time_diff<=max_time_diff)
print(f"{int(np.sum(Data_m2.time_diff>max_time_diff))} points fail max_time_diff")

#remove wakes
Data_trp['waked']=waked['Site 3.2'].interp(time=Data_trp.time)
f_trp=Data_trp[var_trp].where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_trp['waked'].sum(dim='turbine')>0))} wake events at Site 3.2 excluded")

Data_met['waked']=waked['M5'].interp(time=Data_met.time)
f_met=Data_met[var_met].where(Data_met['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_met['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")

Data_m2['waked']=waked['M5'].interp(time=Data_met.time)
f_m2=Data_m2[var_met].where(Data_m2['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_m2['waked'].sum(dim='turbine')>0))} wake events at M2 excluded")

#remove outliers
f_trp=f_trp.where(f_trp>=min_f).where(f_trp<=max_f)
f_met=f_met.where(f_met>=min_f).where(f_met<=max_f)
f_m2=f_m2.where(f_m2>=min_f).where(f_m2<=max_f)

#align nans
real=np.isnan(f_trp).sum(dim='height')+np.isnan(f_met).sum(dim='height')+np.isnan(f_m2).sum(dim='height')==0
f_trp=f_trp.where(real)
f_met=f_met.where(real)
f_m2=f_m2.where(real)

#bias correction
f_trp_bc=f_trp-Data_trp.bias

#Ri
Ri=met.Ri_3_122.interp(time=Data_trp.time)


#stats TROPoe
f_avg=np.zeros((len(f_trp.height),len(bin_Ri)-1))
f_low=np.zeros((len(f_trp.height),len(bin_Ri)-1))
f_top=np.zeros((len(f_trp.height),len(bin_Ri)-1))
for i_h in range(len(f_trp.height)):
    f_sel=f_trp.isel(height=i_h).values
    real=~np.isnan(Ri.values+f_sel)
    f_avg[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),
                                        bins=bin_Ri)[0]
    f_low[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100), 
                                        bins=bin_Ri)[0]
    f_top[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100), 
                                        bins=bin_Ri)[0]
    
trp_stats=xr.Dataset()
trp_stats['f_avg']=xr.DataArray(data=f_avg,coords={'height':f_trp.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
trp_stats['f_low']=xr.DataArray(data=f_low,coords={'height':f_trp.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
trp_stats['f_top']=xr.DataArray(data=f_top,coords={'height':f_trp.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})

#stats TROPoe bias corrected
f_avg=np.zeros((len(f_trp.height),len(bin_Ri)-1))
f_low=np.zeros((len(f_trp.height),len(bin_Ri)-1))
f_top=np.zeros((len(f_trp.height),len(bin_Ri)-1))
for i_h in range(len(f_trp.height)):
    f_sel=f_trp_bc.isel(height=i_h).values
    real=~np.isnan(Ri.values+f_sel)
    f_avg[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),
                                        bins=bin_Ri)[0]
    f_low[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100), 
                                        bins=bin_Ri)[0]
    f_top[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100), 
                                        bins=bin_Ri)[0]
        
trp_stats['f_bc_avg']=xr.DataArray(data=f_avg,coords={'height':f_trp.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
trp_stats['f_bc_low']=xr.DataArray(data=f_low,coords={'height':f_trp.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
trp_stats['f_bc_top']=xr.DataArray(data=f_top,coords={'height':f_trp.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})

#stats met tower
f_avg=np.zeros((len(f_met.height),len(bin_Ri)-1))
f_low=np.zeros((len(f_met.height),len(bin_Ri)-1))
f_top=np.zeros((len(f_met.height),len(bin_Ri)-1))
for i_h in range(len(f_met.height)):
    f_sel=f_met.isel(height=i_h).values
    real=~np.isnan(Ri.values+f_sel)
    f_avg[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),
                                        bins=bin_Ri)[0]
    f_low[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100), 
                                        bins=bin_Ri)[0]
    f_top[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100), 
                                        bins=bin_Ri)[0]
met_stats=xr.Dataset()
met_stats['f_avg']=xr.DataArray(data=f_avg,coords={'height':f_met.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
met_stats['f_low']=xr.DataArray(data=f_low,coords={'height':f_met.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
met_stats['f_top']=xr.DataArray(data=f_top,coords={'height':f_met.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})

#stats M2
f_avg=np.zeros((len(f_m2.height),len(bin_Ri)-1))
f_low=np.zeros((len(f_m2.height),len(bin_Ri)-1))
f_top=np.zeros((len(f_m2.height),len(bin_Ri)-1))
for i_h in range(len(f_m2.height)):
    f_sel=f_m2.isel(height=i_h).values
    real=~np.isnan(Ri.values+f_sel)
    f_avg[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),
                                        bins=bin_Ri)[0]
    f_low[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100), 
                                        bins=bin_Ri)[0]
    f_top[i_h,:]=stats.binned_statistic(Ri.values[real],f_sel[real],
                                        statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100), 
                                        bins=bin_Ri)[0]
    
m2_stats=xr.Dataset()
m2_stats['f_avg']=xr.DataArray(data=f_avg,coords={'height':f_m2.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
m2_stats['f_low']=xr.DataArray(data=f_low,coords={'height':f_m2.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
m2_stats['f_top']=xr.DataArray(data=f_top,coords={'height':f_m2.height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})

#%% Plots
plt.close("all")

#average profiles
plt.figure(figsize=(18,4))

ctr=1
for s in stab_names:
    i_Ri=stab_names[s]
    plt.subplot(1,len(stab_names),ctr)
    plt.plot(met_stats.f_avg.isel(Ri=i_Ri),met_stats.height,'.-k',label='Met (M5)')
    plt.fill_betweenx(met_stats.height,met_stats.f_low.isel(Ri=i_Ri),
                                      met_stats.f_top.isel(Ri=i_Ri),
                                      color='k',alpha=0.25)
    
    plt.plot(m2_stats.f_avg.isel(Ri=i_Ri),m2_stats.height,'.-g',label='Met (M2)')
    plt.fill_betweenx(m2_stats.height,m2_stats.f_low.isel(Ri=i_Ri),
                                      m2_stats.f_top.isel(Ri=i_Ri),
                                      color='g',alpha=0.25)
    
    plt.plot(trp_stats.f_avg.isel(Ri=i_Ri),trp_stats.height,'.-r',label='TROPoe')
    plt.fill_betweenx(trp_stats.height,trp_stats.f_low.isel(Ri=i_Ri),
                                      trp_stats.f_top.isel(Ri=i_Ri),
                                      color='r',alpha=0.25)
    
    plt.plot(trp_stats.f_bc_avg.isel(Ri=i_Ri),trp_stats.height,'.-b',label='TROPoe (bias-corrected)')
    plt.fill_betweenx(trp_stats.height,trp_stats.f_bc_low.isel(Ri=i_Ri),
                                      trp_stats.f_bc_top.isel(Ri=i_Ri),
                                      color='b',alpha=0.25)
    
    plt.plot(-g/cp*f_trp.height+met_stats.f_avg.isel(Ri=i_Ri).isel(height=0),trp_stats.height,'--k')
    
    plt.xlim([15,25])
    plt.grid()
    plt.xlabel(r'$T$ [$^\circ$C]')
    if i_Ri==0:
        plt.ylabel(r'$z$ [m]')
    
    plt.title(s)
    
    
    ctr+=1
plt.tight_layout()
plt.legend()


#average profiles
plt.figure(figsize=(18,4))

ctr=1
for s in stab_names:
    i_Ri=stab_names[s]
    plt.subplot(1,len(stab_names),ctr)
    plt.plot(met_stats.f_avg.isel(Ri=i_Ri),met_stats.height,'.-k',label='Met (M5)')
    plt.fill_betweenx(met_stats.height,met_stats.f_low.isel(Ri=i_Ri),
                                      met_stats.f_top.isel(Ri=i_Ri),
                                      color='k',alpha=0.25)
    
    plt.plot(m2_stats.f_avg.isel(Ri=i_Ri),m2_stats.height,'.-g',label='Met (M2)')
    plt.fill_betweenx(m2_stats.height,m2_stats.f_low.isel(Ri=i_Ri),
                                      m2_stats.f_top.isel(Ri=i_Ri),
                                      color='g',alpha=0.25)
    
    plt.plot(trp_stats.f_avg.isel(Ri=i_Ri),trp_stats.height,'.-r',label='TROPoe')
    plt.fill_betweenx(trp_stats.height,trp_stats.f_low.isel(Ri=i_Ri),
                                      trp_stats.f_top.isel(Ri=i_Ri),
                                      color='r',alpha=0.25)
    
    plt.plot(-g/cp*f_trp.height+met_stats.f_avg.isel(Ri=i_Ri).isel(height=0),trp_stats.height,'--k')
    
    plt.xlim([15,25])
    plt.grid()
    plt.xlabel(r'$T$ [$^\circ$C]')
    if i_Ri==0:
        plt.ylabel(r'$z$ [m]')
    
    plt.title(s)
    
    
    ctr+=1
plt.tight_layout()
plt.legend()
    
