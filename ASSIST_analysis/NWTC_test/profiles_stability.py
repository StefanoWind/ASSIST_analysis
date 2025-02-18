# -*- coding: utf-8 -*-
"""
Cluster profiles by atmospheric stability
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import sys
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import yaml
import pandas as pd
from scipy.stats import norm
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
source_config=os.path.join(cd,'configs','config.yaml')

# source_stb=os.path.join(cd,'data/nwtc.m5.b1/*csv')
time_offset=np.timedelta64(300, 's')
height_sonic=[15,41,61,74,100,119]#[m] sonic heights

#user
unit='ASSIST11'
met='M5'

var='temperature_rec'

#dataset
sources_trp={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c2/*nc',
             'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
             'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}

sources_met={'M5':'data/nwtc.m5.a0/*nc',
             'M2':'data/nwtc.m2.a0/*nc'}

source_stb='data/nwtc.m2.a0/*nc'

g=9.81#[m/s^2]
cp=1005#[J/KgK]


height_assist=1#[m] height of TROPoe's first point

#stats
max_height=0.2#[km] max height in TROPoe
max_L=2000
stb_class={'VS':[0.25,10],
           ' S':[0.01,0.25],
           ' N':[-0.01,0.01],
           ' U':[-0.25,-0.01],
           'VU':[-10,-0.25]}

p_value=0.05

#graphics
cmap = plt.get_cmap("coolwarm")

#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

#load L data
Data_stb= xr.open_mfdataset(source_stb)

name_save_trp=os.path.join(cd,f'data/{unit}_all.nc')
name_save_met=os.path.join(cd,f'data/{met}_int_{unit}_all.nc')

#graphics
colors=[cmap(val) for val in np.linspace(0, 1, len(stb_class))]

#%% Main

if not os.path.isfile(name_save_trp):

    #load tropoe data
    files=glob.glob(os.path.join(cd,sources_trp[unit]))
    Data_trp=xr.open_mfdataset(files).sel(height=slice(0,max_height))
    
    #qc tropoe data
    Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()#remove clouds with low lwp
    
    qc_gamma=Data_trp['gamma']<=config['max_gamma']
    qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
    qc_cbh=Data_trp['height']<=Data_trp['cbh']
    qc=qc_gamma*qc_rmsa*qc_cbh
    Data_trp['temperature_qc']=Data_trp['temperature'].where(qc)#filter temperature
    Data_trp['waterVapor_qc']=  Data_trp['waterVapor'].where(qc)#filter mixing ratio
        
    print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter')
    print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter')
    print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter')
    
    Data_trp=Data_trp['temperature']
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+height_assist)
    
    Data_trp.to_netcdf(name_save_trp)
    Data_trp.close()
    
#load data
Data_trp=xr.open_dataset(name_save_trp)

if not os.path.isfile(name_save_met):
    #load met data
    files=glob.glob(os.path.join(cd,sources_met[met]))
    
    Data_met=xr.open_mfdataset(files)
    
    if "air_temp_rec" in Data_met.data_vars:
        Data_met=Data_met.rename({"air_temp":"temperature"}).rename({"air_temp_rec":"temperature_rec"})
        
    #interpolation
    Data_met=Data_met.interp(time=Data_trp.time)
    
    Data_met.to_netcdf(name_save_met)
    Data_met.close()

#load data
Data_met=xr.open_dataset(name_save_met)

#interpilation
Data_stb=Data_stb.interp(time=Data_trp.time)

Data_stb['class']=xr.DataArray(data=['nn']*len(Data_stb.time),coords={'time':Data_stb.time})
Ri=Data_stb.Ri
for sc in stb_class:
    Data_stb['class'].loc[(Ri>=stb_class[sc][0])*(Ri<=stb_class[sc][1])]=sc

#stats
T_trp_avg=np.zeros((len(Data_trp.height),len(stb_class)))
T_trp_low=np.zeros((len(Data_trp.height),len(stb_class)))
T_trp_top=np.zeros((len(Data_trp.height),len(stb_class)))
for i_sc in range(len(stb_class)):
    sc=list(stb_class.keys())[i_sc]
    for i_h in range(len(Data_trp.height)):
        T_sel=Data_trp.temperature.isel(height=i_h).where(Data_stb['class']==sc).values
        T_trp_avg[i_h,i_sc]=utl.filt_stat(T_sel,np.nanmean)
        T_trp_low[i_h,i_sc]=utl.filt_BS_stat(T_sel,np.nanmean,p_value/2*100)
        T_trp_top[i_h,i_sc]=utl.filt_BS_stat(T_sel,np.nanmean,(1-p_value/2)*100)

Data_trp['temperature_avg']=xr.DataArray(data=T_trp_avg,coords={'height':Data_trp.height,'_class':list(stb_class.keys())})
Data_trp['temperature_low']=xr.DataArray(data=T_trp_low,coords={'height':Data_trp.height,'_class':list(stb_class.keys())})
Data_trp['temperature_top']=xr.DataArray(data=T_trp_top,coords={'height':Data_trp.height,'_class':list(stb_class.keys())})

T_met_avg=np.zeros((len(Data_met.height),len(stb_class)))
T_met_low=np.zeros((len(Data_met.height),len(stb_class)))
T_met_top=np.zeros((len(Data_met.height),len(stb_class)))
for i_sc in range(len(stb_class)):
    sc=list(stb_class.keys())[i_sc]
    for i_h in range(len(Data_met.height)):
        T_sel=Data_met[var].isel(height=i_h).where(Data_stb['class']==sc).values
        T_met_avg[i_h,i_sc]=utl.filt_stat(T_sel,np.nanmean)
        T_met_low[i_h,i_sc]=utl.filt_BS_stat(T_sel,np.nanmean,p_value/2*100)
        T_met_top[i_h,i_sc]=utl.filt_BS_stat(T_sel,np.nanmean,(1-p_value/2)*100)

Data_met['temperature_avg']=xr.DataArray(data=T_met_avg,coords={'height':Data_met.height,'_class':list(stb_class.keys())})
Data_met['temperature_low']=xr.DataArray(data=T_met_low,coords={'height':Data_met.height,'_class':list(stb_class.keys())})
Data_met['temperature_top']=xr.DataArray(data=T_met_top,coords={'height':Data_met.height,'_class':list(stb_class.keys())})

#%% Plots
plt.close("all")

#Ri histograms
ctr=0
plt.figure()
for sc in stb_class:
    plt.hist(Data_stb.Ri.where(Data_stb['class']==sc),bins=np.linspace(stb_class[sc][0],stb_class[sc][1],20),density=True,color=colors[ctr])
    plt.text(-7.5,10+ctr*10,f'{sc.strip()}: {int(np.sum(Data_stb["class"]==sc))} points',color=colors[ctr])
    ctr+=1
plt.grid()
plt.ylabel('Occurrence')
plt.xlabel('Ri')
plt.gca().set_xscale('symlog',linthresh=0.01)
plt.gca().set_yscale('log')
plt.xticks(np.unique(np.array([stb_class[sc] for sc in stb_class])))
   
#average profiles
plt.figure(figsize=(18,4))

for i_sc in range(len(stb_class)):
    plt.subplot(1,len(stb_class),i_sc+1)
    plt.plot(Data_met.temperature_avg.isel(_class=i_sc),Data_met.height,'.-k')
    plt.fill_betweenx(Data_met.height,Data_met.temperature_low.isel(_class=i_sc),
                                      Data_met.temperature_top.isel(_class=i_sc),
                                      color='k',alpha=0.25)
    plt.plot(Data_trp.temperature_avg.isel(_class=i_sc),Data_trp.height,'.-r')
    plt.fill_betweenx(Data_trp.height,Data_trp.temperature_low.isel(_class=i_sc),
                                      Data_trp.temperature_top.isel(_class=i_sc),
                                      color='r',alpha=0.25)
    plt.plot(-g/cp*Data_trp.height+Data_met.temperature_avg.isel(_class=i_sc).isel(height=0),Data_trp.height,'--k')
    
    plt.xlim([15,25])
    plt.grid()
 
