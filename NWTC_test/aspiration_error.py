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
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs

#dataset
source_m5=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats
source_waked=os.path.join(cd,'data/turbine_wakes.nc')#turbine wakes source
source_m2=os.path.join(cd,'data','nwtc.m2.b0.20220420.20220825.csv')#source of M2 data
height_m2=[2,50,80]#[m] M2 heights
time_offset_m2=7#[h] time offset of M2 data

m2_rename={'Temperature @ {h}m [deg C]':'temperature',
           'Avg Wind Speed @ {h}m [m/s]':'ws',
           'Avg Wind Direction @ {h}m [deg]':'wd',
           'Aspirator Fan Speed @ {h}m [RPM]':'rmp',
           'Global Horizontal [W/m^2]':'ghi',
           'Richardson Number (2-80m)':'Ri_2_80'}

g=9.81#[m/s^2] gravity acceleration
cp=1005#[J/KgK] #air heat capacity
rho=1.225
max_T=40#[C] min value
min_T=-5#[C] max value
max_time_diff=10#[s] maximum time difference
height_corr=[38,87,122]
height_fit=38

#user
unit='ASSIST11'#assist id
var_trp='temperature'#selected temperature variable in TROPoe data
var_met='temperature'#selected temperature variable in M5 data

#stats
bin_Ri=np.array([-100,-0.25,-0.03,0.03,0.25,100])#bins in Ri [mix of Hamilton 2019 and Aitken 2014]
p_value=0.05 #p-value for c.i.
perc_lim=[5,95] #[%] percentile limits
N_bins_ws=10
bins_ghi=np.array([-10,0,500,800,1250])
dscaling=1.25*10**(-4)
min_count=5
min_ghi=50
min_ws=0.1

#graphics
stab_names={'S':4,'NS':3,'N':2,'NU':1,'U':0}

#%% Initialization

#read met data
files=glob.glob(source_m5)
Data_m5=xr.open_mfdataset(files)

#read M2 data
Data_m2_df=pd.read_csv(source_m2, parse_dates=[["DATE (MM/DD/YYYY)", "MST"]])
time_m2=np.array([np.datetime64(t) for t in pd.to_datetime(Data_m2_df['DATE (MM/DD/YYYY)_MST'])])+np.timedelta64(time_offset_m2, 'h')
Data_m2=xr.Dataset()
for var in m2_rename:
    if '{h}' in var:
        f=np.zeros((len(time_m2),len(height_m2)))
        for i_h in range(len(height_m2)):
           f[:,i_h]=Data_m2_df[var.format(h=height_m2[i_h])].values
           
        Data_m2[m2_rename[var]]=xr.DataArray(f,coords={'time':time_m2,'height':height_m2})
    else:
        f=Data_m2_df[var].values
        Data_m2[m2_rename[var]]=xr.DataArray(f,coords={'time':time_m2})

Data_m2=Data_m2.resample(time="10min").mean()
Data_m2['time']=Data_m2.time+np.timedelta64(300,'s')


#read wake data
waked=xr.open_dataset(source_waked)

#%% Main

#height interpolation
Data_m2=Data_m2.interp(height=Data_m5.height,time=Data_m5.time)

#QC
Data_m2['temperature']=Data_m2['temperature'].where(Data_m2['rmp']>0)
print(f"{int(np.sum(Data_m2.rmp<=0))} points fail aspiration check")

# #remove wakes
Data_m5['waked']=waked['M5'].interp(time=Data_m5.time)
T_m5=Data_m5['air_temp_rec'].where(Data_m5['waked'].sum(dim='turbine')==0)
print(f"{int(np.sum(Data_m5['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")

Data_m2['waked']=waked['M2'].interp(time=Data_m2.time)
T_m2=Data_m2['temperature'].where(Data_m2['waked'].sum(dim='turbine')==0)
print(f"{int(np.sum(Data_m2['waked'].sum(dim='turbine')>0))} wake events at M2 excluded")

#remove outliers
T_m5=T_m5.where(T_m5>=min_T).where(T_m5<=max_T)
T_m2=  T_m2.where(T_m2>=min_T).where(T_m2<=max_T)

#stats
ws=Data_m5.ws.sel(height=height_fit).values
ghi=Data_m2.ghi.values
dT=(T_m5.sel(height=height_fit)-T_m2.sel(height=height_fit)).values
real=~np.isnan(ws+ghi+dT)
bins_ws=np.nanpercentile(ws, np.linspace(0,perc_lim[1],N_bins_ws+1))
ws_avg=(bins_ws[1:]+bins_ws[:-1])/2
dT_avg=stats.binned_statistic_2d(ws, ghi, dT,statistic=lambda x: utl.filt_stat(x,np.nanmean,perc_lim=perc_lim),bins=[bins_ws,bins_ghi])[0]
dT_low=stats.binned_statistic_2d(ws, ghi, dT,statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),bins=[bins_ws,bins_ghi])[0]
dT_top=stats.binned_statistic_2d(ws, ghi, dT,statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=[bins_ws,bins_ghi])[0]

#scaling
scaling=ghi/(ws*(273.15+T_m5.sel(height=height_fit).values)*rho*cp)
scaling[ghi<=min_ghi]=np.nan
scaling[ws<=min_ws]=np.nan
real=~np.isnan(scaling+dT)
print(f'Correlation between scaling and temperature error: {np.corrcoef(scaling[real],dT[real])[0][1]}')

bins_scaling=np.arange(0,np.nanpercentile(scaling,perc_lim[1])+dscaling,dscaling)
scaling_avg=(bins_scaling[1:]+bins_scaling[:-1])/2
dT_binned=stats.binned_statistic(scaling[real],dT[real],statistic='mean',bins=bins_scaling)[0]
count=stats.binned_statistic(scaling,dT,statistic='count',bins=bins_scaling)[0]
dT_binned[count<min_count]=np.nan
real=~np.isnan(dT_binned)
LF=np.polyfit(scaling_avg[real], dT_binned[real], 1)

#correction
Data_m5_corr=Data_m5.copy()
T_all=Data_m5['air_temp_rec'].values
for h in height_corr:
    i_h=np.where(Data_m5.height.values==h)[0][0]
    X=ghi/(Data_m5.ws.sel(height=h)*(273.15+Data_m5['air_temp_rec'].sel(height=h))*rho*cp)
    corr=(LF[0]*X+LF[1]).values
    corr[ghi<min_ghi]=0
    corr[Data_m5.ws.sel(height=h)<min_ws]=np.nan
    T_all[:,i_h]=Data_m5['air_temp_rec'].sel(height=h).values-corr

Data_m5_corr['air_temp_rec']=xr.DataArray(T_all,coords=T_m5.coords)

T_m5_corr=Data_m5_corr['air_temp_rec'].where(Data_m5_corr['waked'].sum(dim='turbine')==0)
print(f"{int(np.sum(Data_m5['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")
T_m5_corr=T_m5_corr.where(T_m5_corr>=min_T).where(T_m5_corr<=max_T)


#stats (corrected)
dT_corr=(T_m5_corr.sel(height=height_fit)-T_m2.sel(height=height_fit)).values
real=~np.isnan(ws+ghi+dT_corr)
dT_corr_avg=stats.binned_statistic_2d(ws, ghi, dT_corr,statistic=lambda x: utl.filt_stat(x,np.nanmean,perc_lim=perc_lim),bins=[bins_ws,bins_ghi])[0]
# dT_low=stats.binned_statistic_2d(ws, ghi, dT,statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),bins=[bins_ws,bins_ghi])[0]
# dT_top=stats.binned_statistic_2d(ws, ghi, dT,statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=[bins_ws,bins_ghi])[0]


#%% Plots
plt.close('all')
plt.figure(figsize=(18,10))
for i_h in range(len(Data_m5.height)):
    ax=plt.subplot(len(Data_m5.height),1,len(Data_m5.height)-i_h)
    plt.plot(T_m5.time,T_m5.isel(height=i_h),'k',markersize=1,label='M5')
    plt.plot(T_m5_corr.time,T_m5_corr.isel(height=i_h),'g',markersize=1,label='M5 corrected')
    plt.plot(T_m2.time,T_m2.isel(height=i_h),'b',markersize=1,label='M2')
    
    if i_h>0:
        ax.set_xticklabels([])
    else:
        plt.legend()
    plt.ylim([-5,35])
    plt.grid()
    plt.text(T_m5.time[0],30,r'$z='+str(Data_m5.height.values[i_h])+'$ m')
    

plt.figure()
cmap=matplotlib.cm.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0,1,len(bins_ghi)-1)]
for i_ghi in range(len(bins_ghi)-1):
    plt.plot(ws_avg,dT_avg[:,i_ghi],'.-',color=colors[i_ghi],label=r'GHI $\in ['+str(bins_ghi[i_ghi])+', '+str(bins_ghi[i_ghi+1])+')$ W m$^{-2}$')
    plt.plot(ws_avg,dT_corr_avg[:,i_ghi],'--',color=colors[i_ghi])
    plt.fill_between(ws_avg, dT_low[:,i_ghi],  dT_top[:,i_ghi],color=colors[i_ghi],alpha=0.25)
# plt.gca().set_facecolor((0,0,1,0.1))
plt.legend()
plt.grid()

plt.figure()
plt.plot(scaling,dT,'.k',alpha=0.05,markersize=5,label='All data')
plt.plot(scaling_avg,dT_binned,'.b',markersize=10,label='Binned')
plt.plot(np.arange(0,5),np.arange(0,5)*LF[0]+LF[1],'r',label=r'$'+str(np.round(LF[1],2))+'+'+str(np.round(LF[0],2))+r'\cdot x$')
plt.xlim([0,4*10**-3])
plt.ylim([-2,2])
plt.legend()
plt.grid()


