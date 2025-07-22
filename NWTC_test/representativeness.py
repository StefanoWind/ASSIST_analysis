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
from matplotlib.ticker import NullFormatter
import warnings
import glob
import yaml
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['savefig.dpi'] = 500

warnings.filterwarnings('ignore')
plt.close('all')

#%% Inputs
source_met_sta=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats
source_config=os.path.join(cd,'configs','config.yaml')
source_waked=os.path.join(cd,'data/turbine_wakes.nc')

#site
units=['ASSIST10','ASSIST11']#assist ids
var_trp='temperature'#selected variable in TROPoe data
var_met='temperature'#selected variable in M5 data
var_sf='D_res_air_temp_rec'#selected structure function variable in M5 data
wd_align={'ASSIST10':225,'ASSIST11':230}#[deg] direction of alignment (met tower based)
spacing= {'ASSIST10':66,'ASSIST11':440}#[m] distance from tower

#stats
p_value=0.05#for CI
max_height=200#[m] for data selection
max_f=40#[C] max threshold of selected variable
min_f=-5#[C] min threshold of selected variable
max_time_diff=10#[s] maximum difference in time between met and TROPoe
wd_lim=10#[deg] maximum misalignment
bin_space=np.arange(0,550,50)#bins of spacing
bin_Ri=np.array([-10,-0.25,-0.01,0.01,0.25,10])#bins in Ri
max_ti=50#[%] maximum TI for Taylor frozen ot be valid

#graphics
stab_names=['U','NU','N','NS','S']
unit_sel='ASSIST11'

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
std_avg={}
std_low={}
std_top={}
std_pred={}

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
    
    Data_met_sta['waked']=waked['M5'].interp(time=Data_met_sta.time)
    Data_met_sta=Data_met_sta.where(Data_met_sta['waked'].sum(dim='turbine')==0)
    print(f"{int(np.sum(Data_met_sta['waked'].sum(dim='turbine')>0))} wake events at M5 (stats) excluded")
    
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
    ti=Data_met_sta.ws_std.interp(time=Data_trp.time)/ws*100
    
    cos_wd=np.cos(np.radians(Data_met_sta.wd)).interp(time=Data_trp.time)
    sin_wd=np.sin(np.radians(Data_met_sta.wd)).interp(time=Data_trp.time)
    wd=np.degrees(np.arctan2(sin_wd,cos_wd))%360
    
    Ri=Data_met_sta.Ri_3_122.interp(time=Data_trp.time)
    
    D_T=Data_met_sta[var_sf].interp(time=Data_trp.time)**0.5
    space_lag=D_T.lag*ws
    
    #alignment
    ang_diff1=((wd - wd_align[unit] + 180) % 360) - 180
    ang_diff2=((wd - wd_align[unit]+180 + 180) % 360) - 180
    sel_aligned=(np.abs(ang_diff1)<wd_lim)+(np.abs(ang_diff2)<wd_lim)
    diff_sel=diff.where(sel_aligned).where(ti<=max_ti)
    D_T_sel=D_T.where(sel_aligned).where(ti<=max_ti)
    
    #bin statistics
    f_avg=np.zeros((len(height),len(bin_space)-1,len(bin_Ri)-1))
    f_low=np.zeros((len(height),len(bin_space)-1,len(bin_Ri)-1))
    f_top=np.zeros((len(height),len(bin_space)-1,len(bin_Ri)-1))
    for i_h in range(len(height)):
        i_s=0
        for s1,s2 in zip(bin_space[:-1],bin_space[1:]):
            sel_s=(space_lag.isel(height=i_h)>=s1)*(space_lag.isel(height=i_h)<s2)
            i_Ri=0
            for Ri1,Ri2 in zip(bin_Ri[:-1],bin_Ri[1:]):
                sel_Ri=(Ri>=Ri1)*(Ri<Ri2)
                f_sel=D_T_sel.isel(height=i_h).where(sel_s*sel_Ri).values.ravel()
                f_avg[i_h,i_s,i_Ri]=utl.filt_stat(f_sel,np.nanmean)
                f_low[i_h,i_s,i_Ri]=utl.filt_BS_stat(f_sel,np.nanmean,p_value/2*100)
                f_top[i_h,i_s,i_Ri]=utl.filt_BS_stat(f_sel,np.nanmean,(1-p_value/2)*100)

                i_Ri+=1
            i_s+=1
        print(f'Structure function at {height[i_h]} m done')
        
    f_avg[np.isnan(f_top-f_low)]=np.nan
    D_avg[unit]=xr.DataArray(f_avg,coords={'height':height,
                                     'space':(bin_space[1:]+bin_space[:-1])/2,
                                     'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
    
    D_low[unit]=xr.DataArray(f_low,coords={'height':height,
                                     'space':(bin_space[1:]+bin_space[:-1])/2,
                                     'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
    
    D_top[unit]=xr.DataArray(f_top,coords={'height':height,
                                     'space':(bin_space[1:]+bin_space[:-1])/2,
                                     'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
                             

    f_std_avg=np.zeros((len(height),len(bin_Ri)-1))
    f_std_low=np.zeros((len(height),len(bin_Ri)-1))
    f_std_top=np.zeros((len(height),len(bin_Ri)-1))      
    for i_h in range(len(height)):
        i_Ri=0
        for Ri1,Ri2 in zip(bin_Ri[:-1],bin_Ri[1:]):
            sel_Ri=(Ri>=Ri1)*(Ri<Ri2)

            f_sel=     diff_sel.isel(height=i_h).where(sel_Ri).values
            sigma_sel=sigma_trp.isel(height=i_h).where(sel_Ri).values
            f_std_avg[i_h,i_Ri]=np.nanstd(f_sel)-np.nanmean(sigma_sel)
            f_std_low[i_h,i_Ri]=utl.filt_BS_stat(f_sel,np.nanstd,p_value/2*100,perc_lim=[0,100])-np.nanmean(sigma_sel)
            f_std_top[i_h,i_Ri]=utl.filt_BS_stat(f_sel,np.nanstd,(1-p_value/2)*100,perc_lim=[0,100])-np.nanmean(sigma_sel)

            i_Ri+=1
        print(f'RMSD at {height[i_h]} m done')
    
    f_std_avg[np.isnan(f_std_top-f_std_low)]=np.nan
    std_avg[unit]=xr.DataArray(f_std_avg,coords={'height':height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
    std_low[unit]=xr.DataArray(f_std_low,coords={'height':height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
    std_top[unit]=xr.DataArray(f_std_top,coords={'height':height,'Ri':(bin_Ri[1:]+bin_Ri[:-1])/2})
    
    std_pred[unit]=D_avg[unit].interp(space=spacing[unit])
    
#%% Plots
plt.close('all')
cmap = plt.get_cmap("coolwarm")
colors = [cmap(i) for i in np.linspace(0,1,len(bin_Ri)-1)]
for unit in units:
    fig=plt.figure(figsize=(18,5))
    for i_h in range(len(height)):
        ax=plt.subplot(1,len(height),i_h+1)
        for i_Ri in range(len(bin_Ri)-1):
           
            ax.axvline(spacing[unit],0,1.5,color='g',linestyle='-',linewidth=20,alpha=0.1)
            shift=(i_Ri-(len(bin_Ri)-2)/2)/2
            plt.plot(spacing[unit]*(1+shift/10),std_avg[unit].isel(height=i_h,Ri=i_Ri),'^',
                     markersize=10, markerfacecolor=colors[i_Ri][:-1]+(0.5,),markeredgecolor=colors[i_Ri],zorder=10)
            plt.errorbar(spacing[unit]*(1+shift/10),std_avg[unit].isel(height=i_h,Ri=i_Ri),
                                      [[std_avg[unit].isel(height=i_h,Ri=i_Ri)-std_low[unit].isel(height=i_h,Ri=i_Ri)],
                                       [std_top[unit].isel(height=i_h,Ri=i_Ri)-std_avg[unit].isel(height=i_h,Ri=i_Ri)]],
                                      color=colors[i_Ri],capsize=5,alpha=0.75,zorder=10)
            plt.plot(D_avg[unit].space,D_avg[unit].space**(1/3)*10**-3*5,'--k')
                    
            plt.plot(D_avg[unit].space,D_avg[unit].isel(height=i_h,Ri=i_Ri),'.-',color=colors[i_Ri],label=stab_names[i_Ri],markersize=7)
            ax.fill_between(D_avg[unit].space,D_low[unit].isel(height=i_h,Ri=i_Ri),D_top[unit].isel(height=i_h,Ri=i_Ri),
                            color=colors[i_Ri],alpha=0.25,zorder=10)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.ylim([0.01,1.5])
        plt.grid()
        plt.xlabel('Distance form met tower [m]')
        if i_h==0:
            plt.ylabel(r'St.dev. of $\Delta T$ [$^\circ$C]')
            plt.legend(draggable=True)
        else:
            ax.yaxis.set_major_formatter(NullFormatter())
  
#summary figure
fig=plt.figure(figsize=(18,5))
for i_h in range(len(height)):
    ax=plt.subplot(1,len(height),i_h+1)
    for i_Ri in range(len(bin_Ri)-1):
        for unit in units:
            ax.axvline(spacing[unit],0,1.5,color='g',linestyle='-',linewidth=20,alpha=0.1)
            shift=(i_Ri-(len(bin_Ri)-2)/2)/2
            plt.plot(spacing[unit]*(1+shift/10),std_avg[unit].isel(height=i_h,Ri=i_Ri),'^',
                     markersize=10, markerfacecolor=colors[i_Ri][:-1]+(0.5,),markeredgecolor=colors[i_Ri],zorder=10)
            plt.errorbar(spacing[unit]*(1+shift/10),std_avg[unit].isel(height=i_h,Ri=i_Ri),
                                      [[std_avg[unit].isel(height=i_h,Ri=i_Ri)-std_low[unit].isel(height=i_h,Ri=i_Ri)],
                                       [std_top[unit].isel(height=i_h,Ri=i_Ri)-std_avg[unit].isel(height=i_h,Ri=i_Ri)]],
                                      color=colors[i_Ri],capsize=5,alpha=0.75,zorder=10)
        plt.plot(D_avg[unit_sel].space,D_avg[unit_sel].space**(1/3)*10**-3*5,'--k')
                
        plt.plot(D_avg[unit_sel].space,D_avg[unit_sel].isel(height=i_h,Ri=i_Ri),'.-',color=colors[i_Ri],label=stab_names[i_Ri],markersize=7)
        ax.fill_between(D_avg[unit_sel].space,D_low[unit_sel].isel(height=i_h,Ri=i_Ri),D_top[unit_sel].isel(height=i_h,Ri=i_Ri),
                        color=colors[i_Ri],alpha=0.25,zorder=10)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim([0.01,1.5])
    plt.grid()
    plt.xlabel('Distance form met tower [m]')
    if i_h==0:
        plt.ylabel(r'St.dev. of $\Delta T$ [$^\circ$C]')
        plt.legend(draggable=True)
    else:
        ax.yaxis.set_major_formatter(NullFormatter())

#predicted vs observed stdev
plt.figure()
x=[]
y=[]
for unit in units:
    x=np.append(x,std_avg[unit].values)
    y=np.append(y,std_pred[unit].values)
    utl.plot_lin_fit(x,y)
plt.xlabel('Observed st.dev. of $\Delta T$ [$^\circ$C]')
plt.ylabel('Predicted spatial st.dev. of $\Delta T$ [$^\circ$C]')
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()