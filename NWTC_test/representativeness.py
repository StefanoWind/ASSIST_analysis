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
units=['ASSIST10','ASSIST11']#assist id
unit_sel='ASSIST11'
var_trp='temperature'#selected variable in TROPoe data
var_met='temperature'#selected variable in M5 data
var_sf='D_air_temp_rec'#selected structure function variable in M5 data
wd_align={'ASSIST10':225,'ASSIST11':230}#[deg] direction of alignment (met tower based)
spacing={'ASSIST10':66,'ASSIST11':440}

#stats
p_value=0.05#for CI
max_height=200#[m]
max_f=40#[C] max threshold of selected variable
min_f=-5#[C] min threshold of selected variable
max_time_diff=10#[s] maximum difference in time between met and TROPoe
wd_lim=10#[deg] maximum misalignment
bin_space=np.arange(0,550,50)#bins of spacing
bin_ws=np.array([0,2,4,6,8,10,12,25])#bins of wind speed
 
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
D_avg={}
D_low={}
D_top={}
rmsd_avg={}
rmsd_low={}
rmsd_top={}

#%% Main

for unit in units:
    #read and align data
    Data_trp=xr.open_dataset(os.path.join(cd,'data',f'tropoe.{unit}.bias.nc'))
    Data_met=xr.open_dataset(os.path.join(cd,'data',f'met.a1.{unit}.nc'))
    Data_trp,Data_met=xr.align(Data_trp,Data_met,join="inner",exclude=["height"])
    
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
    
    D_T=Data_met_sta[var_sf].interp(time=Data_trp.time)**0.5
    space_lag=D_T.lag*ws
    
    #alignment
    ang_diff1=((wd - wd_align[unit] + 180) % 360) - 180
    ang_diff2=((wd - wd_align[unit]+180 + 180) % 360) - 180
    sel_aligned=(np.abs(ang_diff1)<wd_lim)+(np.abs(ang_diff2)<wd_lim)
    diff_sel=diff.where(sel_aligned)
    D_T_sel=D_T.where(sel_aligned)
    
    #bin statistics
    f_avg=np.zeros((len(height),len(bin_space)-1,len(bin_ws)-1))
    f_low=np.zeros((len(height),len(bin_space)-1,len(bin_ws)-1))
    f_top=np.zeros((len(height),len(bin_space)-1,len(bin_ws)-1))
    for i_h in range(len(height)):
        i_s=0
        for s1,s2 in zip(bin_space[:-1],bin_space[1:]):
            sel_s=(space_lag.isel(height=i_h)>=s1)*(space_lag.isel(height=i_h)<s2)
            i_ws=0
            for ws1,ws2 in zip(bin_ws[:-1],bin_ws[1:]):
                sel_ws=(ws.isel(height=i_h)>=ws1)*(ws.isel(height=i_h)<ws2)
                f_sel=D_T_sel.isel(height=i_h).where(sel_s*sel_ws).values.ravel()
                f_avg[i_h,i_s,i_ws]=utl.filt_stat(f_sel,np.nanmean)
                f_low[i_h,i_s,i_ws]=utl.filt_BS_stat(f_sel,np.nanmean,p_value/2*100)
                f_top[i_h,i_s,i_ws]=utl.filt_BS_stat(f_sel,np.nanmean,(1-p_value/2)*100)
                i_ws+=1
            i_s+=1
        print(f'Structure function at {height[i_h]} m done')
        
    f_avg[np.isnan(f_top-f_low)]=np.nan
    D_avg[unit]=xr.DataArray(f_avg,coords={'height':height,
                                     'space':(bin_space[1:]+bin_space[:-1])/2,
                                     'ws':(bin_ws[1:]+bin_ws[:-1])/2})
    
    D_low[unit]=xr.DataArray(f_low,coords={'height':height,
                                     'space':(bin_space[1:]+bin_space[:-1])/2,
                                     'ws':(bin_ws[1:]+bin_ws[:-1])/2})
    
    D_top[unit]=xr.DataArray(f_top,coords={'height':height,
                                     'space':(bin_space[1:]+bin_space[:-1])/2,
                                     'ws':(bin_ws[1:]+bin_ws[:-1])/2})
    
    f_rmsd_avg=np.zeros((len(height),len(bin_ws)-1)) 
    f_rmsd_low=np.zeros((len(height),len(bin_ws)-1)) 
    f_rmsd_top=np.zeros((len(height),len(bin_ws)-1))        
    for i_h in range(len(height)):
        i_ws=0
        for ws1,ws2 in zip(bin_ws[:-1],bin_ws[1:]):
            sel_ws=(ws.isel(height=i_h)>=ws1)*(ws.isel(height=i_h)<ws2)
            f_sel=diff_sel.isel(height=i_h).where(sel_ws).values
            f_rmsd_avg[i_h,i_ws]=np.nanmean(f_sel**2)**0.5
            f_rmsd_low[i_h,i_ws]=utl.filt_BS_stat((f_sel**2),np.nanmean,p_value/2*100,perc_lim=[0,100])**0.5
            f_rmsd_top[i_h,i_ws]=utl.filt_BS_stat((f_sel**2),np.nanmean,(1-p_value/2)*100,perc_lim=[0,100])**0.5
            i_ws+=1
        print(f'RMSD at {height[i_h]} m done')
    
    f_rmsd_avg[np.isnan(f_rmsd_top-f_rmsd_low)]=np.nan
    rmsd_avg[unit]=xr.DataArray(f_rmsd_avg,coords={'height':height,'ws':(bin_ws[1:]+bin_ws[:-1])/2})
    rmsd_low[unit]=xr.DataArray(f_rmsd_low,coords={'height':height,'ws':(bin_ws[1:]+bin_ws[:-1])/2})
    rmsd_top[unit]=xr.DataArray(f_rmsd_top,coords={'height':height,'ws':(bin_ws[1:]+bin_ws[:-1])/2})
    
    
#%% Plots
for unit in units:
    fig=plt.figure(figsize=(16,5))
    colors = [cmap(i) for i in np.linspace(0,1,len(height))]
    for i_ws in range(len(D_avg[unit].ws)):
        ax=plt.subplot(1,len(D_avg[unit].ws),i_ws+1)
        for i_h in range(len(height)):
            ax.axvline(spacing[unit],0,1.5,color='k',linestyle='--')
            plt.plot(D_avg[unit].space,D_avg[unit].isel(height=i_h,ws=i_ws),'.-',color=colors[i_h],label=r'$z='+str(int(height[i_h]))+'$ m')
            ax.fill_between(D_avg[unit].space,D_low[unit].isel(height=i_h,ws=i_ws),D_top[unit].isel(height=i_h,ws=i_ws),color=colors[i_h],alpha=0.25)
            plt.plot(spacing[unit]+(i_h-(len(height)-1)/2)*10,rmsd_avg[unit].isel(height=i_h,ws=i_ws),'^',color=colors[i_h],alpha=0.75)
            plt.errorbar(spacing[unit]+(i_h-(len(height)-1)/2)*10,rmsd_avg[unit].isel(height=i_h,ws=i_ws),
                                      [[rmsd_avg[unit].isel(height=i_h,ws=i_ws)-rmsd_low[unit].isel(height=i_h,ws=i_ws)],
                                       [rmsd_top[unit].isel(height=i_h,ws=i_ws)-rmsd_avg[unit].isel(height=i_h,ws=i_ws)]],
                                      color=colors[i_h],capsize=5,alpha=0.75)
            
        plt.ylim([0,1.5])
        plt.grid()
        plt.xlabel('Spacing [m]')
        if i_ws==0:
            plt.ylabel(r'RMSD of $\Delta T$ [$^\circ$C] ('+unit+')')
            plt.legend()

#summary figure
fig=plt.figure(figsize=(16,5))
colors = [cmap(i) for i in np.linspace(0,1,len(height))]
for i_ws in range(len(D_avg[unit].ws)):
    ax=plt.subplot(1,len(D_avg[unit].ws),i_ws+1)
    for i_h in range(len(height)):
        for unit in units:
            ax.axvline(spacing[unit],0,1.5,color='k',linestyle='--')
            plt.plot(spacing[unit]+(i_h-(len(height)-1)/2)*10,rmsd_avg[unit].isel(height=i_h,ws=i_ws),'^',color=colors[i_h],alpha=0.75)
            plt.errorbar(spacing[unit]+(i_h-(len(height)-1)/2)*10,rmsd_avg[unit].isel(height=i_h,ws=i_ws),
                                      [[rmsd_avg[unit].isel(height=i_h,ws=i_ws)-rmsd_low[unit].isel(height=i_h,ws=i_ws)],
                                       [rmsd_top[unit].isel(height=i_h,ws=i_ws)-rmsd_avg[unit].isel(height=i_h,ws=i_ws)]],
                                      color=colors[i_h],capsize=5,alpha=0.75)
                
        plt.plot(D_avg[unit_sel].space,D_avg[unit_sel].isel(height=i_h,ws=i_ws),'.-',color=colors[i_h],label=r'$z='+str(int(height[i_h]))+'$ m')
        ax.fill_between(D_avg[unit_sel].space,D_low[unit_sel].isel(height=i_h,ws=i_ws),D_top[unit_sel].isel(height=i_h,ws=i_ws),color=colors[i_h],alpha=0.25)
        

    plt.ylim([0,1.5])
    plt.grid()
    plt.xlabel('Distance form met tower [m]')
    if i_ws==0:
        plt.ylabel(r'RMSD of $\Delta T$ [$^\circ$C]')
        plt.legend()


S,WS=np.meshgrid(D_avg[unit_sel].space,D_avg[unit_sel].ws)

fig=plt.figure(figsize=(16,5))
for i_h in range(len(height)):
    ax=plt.subplot(1,len(height),i_h+1,projection='3d')
    ax.plot_surface(S,WS,D_avg[unit_sel].isel(height=i_h).T,cmap='plasma',linewidth=0)
    for unit in units:
        ax.plot(spacing[unit]+np.zeros(len(rmsd_avg[unit].ws)),rmsd_avg[unit].ws,rmsd_avg[unit].isel(height=i_h),'.g',markersize=15, zorder=10)
    
    ax.view_init(azim=45,elev=25)
    ax.set_zlim([0,1.5])