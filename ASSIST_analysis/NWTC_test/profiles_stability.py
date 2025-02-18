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

source_stb=os.path.join(cd,'data/nwtc.m5.b1/*csv')
time_offset=np.timedelta64(300, 's')
height_sonic=[15,41,61,74,100,119]#[m] sonic heights

#user
unit='ASSIST10'
met='M5'

var='temperature_rec'

#dataset
sources_trp={'ASSIST10':'data/awaken/nwtc.assist.tropoe.z01.c2/*nc',
             'ASSIST11':'data/awaken/nwtc.assist.tropoe.z02.c0/*nc',
             'ASSIST12':'data/awaken/nwtc.assist.tropoe.z03.c0/*nc'}

sources_met={'M5':'data/nwtc.m5.a0/*nc',
             'M2':'data/nwtc.m2.a0/*nc'}


height_assist=1#[m] height of TROPoe's first point

#stats
max_height=0.2#[km] max height in TROPoe
max_L=2000
stb_class={'U':[-500,0],
           'S':[0,500],
           'N1':[-max_L,-500],
           'N2':[500,max_L]}
stb_class_uni=['S','N','U']


height_sel=74#[m]


#%% Initialization
#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

#load L data
files=glob.glob(source_stb)
data_stb= pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

time=data_stb.iloc[:,0].values*np.timedelta64(1, 's')+np.datetime64('1970-01-01T00:00:00')+time_offset

L=np.zeros((len(time),len(height_sonic)))
for i_h in range(len(height_sonic)):
    L[:,i_h]=data_stb[f'MO_Length_Sonic_{height_sonic[i_h]}m (m)'].values
    
Data_stb=xr.Dataset()
Data_stb['L']=xr.DataArray(data=L,coords={'time':time,'height':height_sonic})

name_save_trp=os.path.join(cd,f'data/{unit}_all.nc')
name_save_met=os.path.join(cd,f'data/{met}_int_{unit}_all.nc')

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

#qc
Data_stb['L']=Data_stb.L.clip(min=-max_L, max=max_L)
print(f'{np.round(np.sum(np.abs(Data_stb.L).values==max_L)/np.sum(~np.isnan(Data_stb.L.values))*100,1)}% excluded L values')

#interpilation
Data_stb=Data_stb.interp(time=Data_trp.time)

Data_stb['class']=xr.DataArray(data=['']*len(Data_stb.time),coords={'time':Data_stb.time})
L=Data_stb.L.sel(height=height_sel)
for sc in stb_class:
    Data_stb['class'].loc[(L>=stb_class[sc][0])*(L<=stb_class[sc][1])]=sc[0]

#stats
T_trp_avg=np.zeros((len(Data_trp.height),len(stb_class_uni)))
T_trp_low=np.zeros((len(Data_trp.height),len(stb_class_uni)))
T_trp_top=np.zeros((len(Data_trp.height),len(stb_class_uni)))
for i_sc in range(len(stb_class_uni)):
    for i_h in range(len(Data_trp.height)):
        T_trp_avg[i_h,i_sc]=utl.filt_stat(Data_trp.temperature.isel(height=i_h).where(Data_stb['class']==stb_class_uni[i_sc]).values,np.nanmean)

Data_trp['temperature_avg']=xr.DataArray(data=T_trp_avg,coords={'height':Data_trp.height,'_class':stb_class_uni})

T_met_avg=np.zeros((len(Data_met.height),len(stb_class_uni)))
T_met_low=np.zeros((len(Data_met.height),len(stb_class_uni)))
T_met_top=np.zeros((len(Data_met.height),len(stb_class_uni)))
for i_sc in range(len(stb_class_uni)):
    for i_h in range(len(Data_met.height)):
        T_met_avg[i_h,i_sc]=utl.filt_stat(Data_met[var].isel(height=i_h).where(Data_stb['class']==stb_class_uni[i_sc]).values,np.nanmean)

Data_met['temperature_avg']=xr.DataArray(data=T_met_avg,coords={'height':Data_met.height,'_class':stb_class_uni})

#%% Plots
plt.close("all")
#L histograms
sel=~np.isnan(Data_stb.L.mean(dim='time'))
hsel=Data_stb.height[sel].values
plt.figure(figsize=(18,10))
for i_h in range(len(hsel)):
    ax=plt.subplot(len(hsel),1,i_h+1)
    plt.hist(Data_stb.L.sel(height=hsel[i_h]),bins=np.linspace(-max_L,max_L,100),color='k')
    plt.grid()
    plt.ylabel('Occurrence')
    if i_h==len(hsel)-1:
        plt.xlabel(r'$L$ [m]')
    else:
        ax.set_xticklabels([])
    plt.text(-max_L*0.9,100,r'$z='+str(hsel[i_h])+'$ m')
plt.tight_layout()
    
#average profiles
plt.figure(figsize=(18,7))
ctr=1
for sc in stb_class_uni:
    plt.subplot(1,len(stb_class_uni),ctr)
    plt.plot(Data_met.temperature_avg.sel(_class=sc),Data_met.height,'.-k')
    plt.plot(Data_trp.temperature_avg.sel(_class=sc),Data_trp.height,'.-r')
    ctr+=1


dT_dz_met=Data_met[var].isel(height=1)-Data_met[var].isel(height=0)
dT_dz_trp=Data_trp['temperature'].isel(height=1)-Data_trp['temperature'].isel(height=0)
DT_avg=np.abs(Data_trp.temperature.interp(height=Data_met.height)-Data_met[var]).mean(dim='height')

plt.figure()
plt.scatter(L,dT_dz_met,s=DT_avg*10)
plt.gca().set_xscale('symlog')
plt.grid()