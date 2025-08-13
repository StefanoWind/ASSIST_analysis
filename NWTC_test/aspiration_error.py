# -*- coding: utf-8 -*-
"""
Evaluate and correct aspiration bias 
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
from matplotlib.ticker import ScalarFormatter
import glob
import warnings
import yaml
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs

#dataset
path_config=os.path.join(cd,'configs/config.yaml')
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

max_T=40#[C] min value
min_T=-5#[C] max value
max_time_diff=301#[s] maximum time difference
height_corr=[38,87,122]#[m] heights at which correction is needed
height_fit=38#[m] height at which fit is evaulated

#user
unit='ASSIST10'#assist id

#stats
p_value=0.05 #p-value for c.i.
perc_lim=[5,95] #[%] percentile limits
N_bins_ws=10#number of bins in wind speed
bins_ghi=np.array([-10,0,500,800,1250])#bins in GHI
dscaling=1.25*10**(-4) #step in binning of scaling parameter [Nakamura nd Mahrt 2005]
min_count=5#minimum points per bin
min_ghi=50#minimum GHI to perform fit
min_ws=0.25 #[m/s] minimum wind speed to perform correction

#%% Functions
def vapor_pressure(Td):
    """
    Partial vapor pressure from dewpoint temperature
    """
    #constants for T>=0 C
    A1=7.5
    B1=237.3
    
    #constants for T<0 C
    A2=9.5
    B2=265.5
    
    if Td.shape==():
        if Td>=0:
            e=6.11*10**((A1*Td)/(Td+B1))*100
        else:
            e=6.11*10**((A2*Td)/(Td+B2))*100
    else:
        e1=6.11*10**((A1*Td)/(Td+B1))*100
        e2=6.11*10**((A2*Td)/(Td+B2))*100
        e=e1.where(Td>=0,e2)
 
    return e

#%% Initialization

with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#read M5 stats data
files=glob.glob(source_m5)
Data_m5=xr.open_mfdataset(files)

#read M5 high-frequency data
Data_m5_hf=xr.open_dataset(os.path.join(cd,'data',f'met.a1.{unit}.nc'))

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

#10-min averaging
Data_m2=Data_m2.resample(time="10min").mean()
Data_m2['time']=Data_m2.time+np.timedelta64(300,'s')

#read wake data
waked=xr.open_dataset(source_waked)

#%% Main

#qc
data=Data_m5.where(Data_m5.precip.isel(height=0)==0)#excluding precipitation
print(f"{int(np.sum(Data_m5.precip.isel(height=0)>0))} precipitation events excluded")

Data_m5_hf=Data_m5_hf.where(np.abs(Data_m5_hf.time_diff)<=max_time_diff)
print(f"{int(np.sum(Data_m5_hf.time_diff>max_time_diff))} points fail max_time_diff")

Data_m2['temperature']=Data_m2['temperature'].where(Data_m2['rmp']>0)
print(f"{int(np.sum(Data_m2.rmp<=0))} points fail aspiration check")

#time vectors
tnum_m5=   (Data_m5.time-   np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
tnum_m2=   (Data_m2.time-   np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
tnum_m5_hf=(Data_m5_hf.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')

#interpolation of M2 into M5 (height and time)
time_diff=tnum_m2.interp(time=Data_m5.time.values,method='nearest')-tnum_m5
Data_m2=Data_m2.interp(height=Data_m5.height,time=Data_m5.time)
Data_m2=Data_m2.where(np.abs(time_diff)<=max_time_diff)
print(f"{int(np.sum(np.abs(time_diff)>=max_time_diff))} points fail max_time_diff for M2->M5")

#interpolation of M5 stats into M5 high-frequency
time_diff=tnum_m5.interp(time=Data_m5_hf.time.values,method='nearest')-tnum_m5_hf
Data_m5_int=Data_m5.interp(time=Data_m5_hf.time)
Data_m5_int=Data_m5_int.where(np.abs(time_diff)<=max_time_diff)
print(f"{int(np.sum(np.abs(time_diff)>=max_time_diff))} points fail max_time_diff for M5 stats->M5 high-frequency")

#interpolation of GHI into M5 high-frequency
time_diff=tnum_m2.interp(time=Data_m5_hf.time.values,method='nearest')-tnum_m5_hf
ghi_int=Data_m2.ghi.interp(time=Data_m5_hf.time)
ghi_int=ghi_int.where(np.abs(time_diff)<=max_time_diff)
print(f"{int(np.sum(np.abs(time_diff)>=max_time_diff))} points fail max_time_diff for GHI")

#remove wakes
Data_m5['waked']=waked['M5'].interp(time=Data_m5.time)
T_m5=Data_m5['air_temp_rec'].where(Data_m5['waked'].sum(dim='turbine')==0)
print(f"{int(np.sum(Data_m5['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")

Data_m2['waked']=waked['M2'].interp(time=Data_m2.time)
T_m2=Data_m2['temperature'].where(Data_m2['waked'].sum(dim='turbine')==0)
print(f"{int(np.sum(Data_m2['waked'].sum(dim='turbine')>0))} wake events at M2 excluded")

#remove outliers
T_m5=T_m5.where(T_m5>=min_T).where(T_m5<=max_T)
T_m2=T_m2.where(T_m2>=min_T).where(T_m2<=max_T)

#dT stats
ws=Data_m5.ws.sel(height=height_fit).values
ghi=Data_m2.ghi.values
dT=(T_m5.sel(height=height_fit)-T_m2.sel(height=height_fit)).values
real=~np.isnan(ws+ghi+dT)
bins_ws=np.nanpercentile(ws, np.linspace(0,perc_lim[1],N_bins_ws+1))
ws_avg=(bins_ws[1:]+bins_ws[:-1])/2
dT_avg=stats.binned_statistic_2d(ws,ghi,dT,statistic=lambda x: utl.filt_stat(x,np.nanmean,perc_lim=perc_lim),                             bins=[bins_ws,bins_ghi])[0]
dT_low=stats.binned_statistic_2d(ws,ghi,dT,statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=[bins_ws,bins_ghi])[0]
dT_top=stats.binned_statistic_2d(ws,ghi,dT,statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=[bins_ws,bins_ghi])[0]

#scaling
scaling=ghi/(ws*(273.15+T_m5.sel(height=height_fit).values)*config['rho']*config['cp'])
scaling[ghi<=min_ghi]=np.nan
scaling[ws<=min_ws]=np.nan
real=~np.isnan(scaling+dT)
print(f'Correlation between scaling and temperature error: {np.corrcoef(scaling[real],dT[real])[0][1]}')

#fit for aspiration error correction
bins_scaling=np.arange(0,np.nanpercentile(scaling,perc_lim[1])+dscaling,dscaling)
scaling_avg=(bins_scaling[1:]+bins_scaling[:-1])/2
dT_binned=stats.binned_statistic(scaling[real],dT[real],statistic= lambda x: utl.filt_stat(x,np.nanmean,perc_lim=perc_lim),bins=bins_scaling)[0]
count=stats.binned_statistic(scaling,dT,statistic='count',bins=bins_scaling)[0]
dT_binned[count<min_count]=np.nan
real=~np.isnan(dT_binned)
LF=np.polyfit(scaling_avg[real], dT_binned[real], 1)

#correction (stats)
Data_m5_corr=Data_m5.copy()
T_all=Data_m5['air_temp_rec'].values+0
for h in height_corr:
    i_h=np.where(Data_m5.height.values==h)[0][0]
    ws_cap=Data_m5.ws.sel(height=h).where(Data_m5.ws.sel(height=h)>min_ws,min_ws)
    X=Data_m2.ghi/(ws_cap*(273.15+Data_m5['air_temp_rec'].sel(height=h))*config['rho']*config['cp'])
    corr=(LF[0]*X+LF[1]).values
    corr[Data_m2.ghi<min_ghi]=0
    T_all[:,i_h]=Data_m5['air_temp_rec'].sel(height=h).values-corr

Data_m5_corr['air_temp_rec']=xr.DataArray(T_all,coords=T_m5.coords)

#Ri correction
#pressure gradient
e=vapor_pressure(Data_m5_corr['dewp_temp'])
P_s=Data_m5_corr['press'].isel(height=0)*100
q_s=0.622*e.isel(height=0)/P_s
Tv_s=(Data_m5_corr['air_temp_rec'].isel(height=0)+273.15)*(1+0.61*q_s)
dP_dz=-config['g']*P_s/(config['R_a']*Tv_s) 

#potential virtual temperature
Data_m5_corr['press']=(P_s+(Data_m5_corr.height-Data_m5_corr.height[0])*dP_dz)/100
q=0.622*e/(Data_m5_corr['press']*100)
Data_m5_corr['Tv']=(Data_m5_corr['air_temp_rec']+273.15)*(1+0.61*q)
Data_m5_corr['theta_v']= Data_m5_corr['Tv']*(config['P_ref']/(Data_m5_corr['press']*100))**(config['R_a']/config['cp'])

#Richardson number
Data_m5_corr['um']=Data_m5_corr['ws']*np.cos(np.radians(270-Data_m5_corr['wd']))
Data_m5_corr['vm']=Data_m5_corr['ws']*np.sin(np.radians(270-Data_m5_corr['wd']))
height=Data_m5_corr.height.values
for h1 in height:
    for h2 in height[height>h1]:
        theta_v_avg=(Data_m5_corr['theta_v'].sel(height=h2)+Data_m5_corr['theta_v'].sel(height=h1))/2
        dtheta_v_dz=(Data_m5_corr['theta_v'].sel(height=h2)-Data_m5_corr['theta_v'].sel(height=h1))/(h2-h1)
        dum_dz=     (Data_m5_corr['um'].sel(height=h2)-     Data_m5_corr['um'].sel(height=h1))/(h2-h1)
        dvm_dz=     (Data_m5_corr['vm'].sel(height=h2)-     Data_m5_corr['vm'].sel(height=h1))/(h2-h1)
        Data_m5_corr[f'Ri_{h1}_{h2}']=config['g']/theta_v_avg*dtheta_v_dz/(dum_dz**2+dvm_dz**2)

#correction (high-frequency)
Data_m5_hf_corr=Data_m5_hf.copy()
T_all=Data_m5_hf['temperature'].values+0
for h in height_corr:
    i_h=np.where(Data_m5_hf_corr.height.values==h)[0][0]
    ws_cap=Data_m5_int.ws.sel(height=h).where(Data_m5_int.ws.sel(height=h)>min_ws,min_ws)
    X=ghi_int/(ws_cap*(273.15+Data_m5_int['air_temp_rec'].sel(height=h))*config['rho']*config['cp'])
    corr=(LF[0]*X+LF[1]).values
    corr[ghi_int<min_ghi]=0
    T_all[:,i_h]=Data_m5_hf['temperature'].sel(height=h).values-corr

Data_m5_hf_corr['temperature']=xr.DataArray(T_all,coords=Data_m5_hf.temperature.coords)

#%% Output
Data_m5_corr.to_netcdf(os.path.join(cd,'data',source_m5.split('/')[-2]+'.corr.nc'))
Data_m5_hf_corr.to_netcdf(os.path.join(cd,'data',f'met.a1.{unit}.corr.nc'))

#%% Plots
plt.close('all')

#all T stats corrected
plt.figure(figsize=(18,10))
for i_h in range(len(Data_m5.height)):
    ax=plt.subplot(len(Data_m5.height),1,len(Data_m5.height)-i_h)
    plt.plot(Data_m5.time,Data_m5.air_temp_rec.isel(height=i_h),'k',label='M5')
    plt.plot(Data_m5_corr.time,Data_m5_corr.air_temp_rec.isel(height=i_h),'g',markersize=1,label='M5 corrected')
    plt.plot(T_m2.time,T_m2.isel(height=i_h),'b',markersize=1,label='M2')
    
    if i_h>0:
        ax.set_xticklabels([])
    else:
        plt.legend()
        plt.xlabel('Time (UTC)')
    plt.ylim([-5,35])
    plt.grid()
    plt.text(T_m5.time[0],30,r'$z='+str(Data_m5.height.values[i_h])+'$ m')
    plt.ylabel(r'$T$ [$^\circ$C]')
   
# all T high-frequency corrected
plt.figure(figsize=(18,10))
for i_h in range(len(Data_m5_hf.height)):
    ax=plt.subplot(len(Data_m5_hf.height),1,len(Data_m5_hf.height)-i_h)
    plt.plot(Data_m5_hf.time,Data_m5_hf.temperature.isel(height=i_h),'k',label='M5')
    plt.plot(Data_m5_hf_corr.time,Data_m5_hf_corr.temperature.isel(height=i_h),'g',markersize=1,label='M5 corrected')
    plt.plot(T_m2.time,T_m2.isel(height=i_h),'b',markersize=1,label='M2')
    
    if i_h>0:
        ax.set_xticklabels([])
    else:
        plt.legend()
        plt.xlabel('Time (UTC)')
    plt.ylim([-5,35])
    plt.grid()
    plt.text(Data_m5_hf.time[0],30,r'$z='+str(Data_m5_hf.height.values[i_h])+'$ m')
    plt.ylabel(r'$T$ [$^\circ$C]')
    
#dT stats
plt.figure(figsize=(18,8))
cmap=matplotlib.cm.get_cmap('plasma')
plt.subplot(1,2,1)
colors = [cmap(i) for i in np.linspace(0,1,len(bins_ghi)-1)]
for i_ghi in range(len(bins_ghi)-1):
    plt.plot(ws_avg,dT_avg[:,i_ghi],'.-',markersize=10,color=colors[i_ghi],label=r'GHI $\in ['+str(bins_ghi[i_ghi])+', '+str(bins_ghi[i_ghi+1])+')$ W m$^{-2}$')
    plt.fill_between(ws_avg, dT_low[:,i_ghi],  dT_top[:,i_ghi],color=colors[i_ghi],alpha=0.25)
plt.legend()
plt.grid()
plt.xlabel(r'$\overline{U}$ [m s$^{-1}$]')
plt.ylabel(r'$\Delta \overline{T}$ (M5-M2) [$^\circ$C]')

#correction
ax=plt.subplot(1,2,2)
plt.plot(scaling,dT,'.k',alpha=0.1,markersize=10,label='All data')
plt.plot(scaling_avg,dT_binned,'.b',markersize=20,label='Binned',zorder=10)
plt.plot([0,2*10**-3],np.array([0,2*10**-3])*LF[0]+LF[1],'r',label=r'$'+str(np.round(LF[1],2))+'+'+str(np.round(LF[0],2))+r'\cdot x$')
plt.xlim([0,2*10**-3])
plt.ylim([-2,2])
plt.legend()
plt.xlabel(r'GHI $(\rho c_p ~\overline{T}~ \overline{U})^{-1}$')
plt.ylabel(r'$\Delta \overline{T}$ (M5-M2) [$^\circ$C]')
plt.grid()
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # always use scientific notation
ax.xaxis.set_major_formatter(formatter)

plt.figure(figsize=(18,6))
plt.plot(Data_m5.time,Data_m5.Ri_3_122,'-k',label='M5')
plt.plot(Data_m5_corr.time,Data_m5_corr.Ri_3_122,'-g',label='M5 (corrected)')
plt.gca().set_yscale('symlog')
plt.ylim([-100,100])
plt.grid(True)
plt.xlabel('Time (UTC)')
plt.ylabel('Ri')
plt.legend()



