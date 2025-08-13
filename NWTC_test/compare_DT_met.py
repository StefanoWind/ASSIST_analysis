# -*- coding: utf-8 -*-
"""
Compare temperature gradients from tropoe retrievals to met tower data
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
import matplotlib.gridspec as gridspec
import glob
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs

#dataset
source_config=os.path.join(cd,'configs','config.yaml')
source_waked=os.path.join(cd,'data/turbine_wakes.nc')
source_met_sta=os.path.join(cd,'data/nwtc.m5.c1.corr.nc')#source of met stats
sigma_met=0.1#[C] uncertaiinty of met measurements [St Martin et al. 2016]
site_trp= {'ASSIST10':'Site 4.0','ASSIST11':'Site 3.2'}
bin_Ri=np.array([-100,-0.25,-0.03,0.03,0.25,100])#bins in Ri [mix of Hamilton 2019 and Aitken 2014]
stab_names={'S':4,'NS':3,'N':2,'NU':1,'U':0}

#user
unit='ASSIST11'#assist id
h2=122#upper height for Ri
h1=3#upper height for Ri

#stats
p_value=0.05#for CI
max_height=200#[m] maximum height
max_T=40#[C] max threshold of selected variable
min_T=-5#[C] min threshold of selected variable
max_time_diff=10#[s] maximum difference in time between met and TROPoe

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
    
    Tdv=Td.values
    e=Tdv+np.nan
    e[Tdv>=0]=6.11*10**((A1*Tdv[Tdv>=0])/(Tdv[Tdv>=0]+B1))*100
    e[Tdv<0]= 6.11*10**((A2*Tdv[Tdv<0])/(Tdv[Tdv<0]+B2))*100
 
    return xr.DataArray(e,coords=Td.coords)

def to_latex(arr, align="c"):
    ncols = arr.shape[1]
    header = "\\begin{tabular}{" + ("|".join([align] * ncols)) + "}"
    lines = [header, "\\hline"]
    for row in arr:
        line = " & ".join(map(str, row)) + r" \\ \hline"
        lines.append(line)
    lines.append("\\end{tabular}")
    return "\n".join(lines)

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

#%% Main

#height interpolation
Data_trp=Data_trp.interp(height=Data_met.height)

#QC
Data_trp=Data_trp.where(Data_trp.qc==0)
print(f"{int(np.sum(Data_trp.qc!=0))} points fail QC in TROPoe")

Data_met=Data_met.where(Data_met.time_diff<=max_time_diff)
print(f"{int(np.sum(Data_met.time_diff>max_time_diff))} points fail max_time_diff")

#remove wake
Data_trp['waked']=waked[site_trp[unit]].interp(time=Data_trp.time)
Data_trp=Data_trp.where(Data_trp['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_trp['waked'].sum(dim='turbine')>0))} wake events at {site_trp[unit]} excluded")

Data_met['waked']=waked['M5'].interp(time=Data_met.time)
Data_met=Data_met.where(Data_met['waked'].sum(dim='turbine')==0).sel(height=slice(0,max_height))
print(f"{int(np.sum(Data_met['waked'].sum(dim='turbine')>0))} wake events at M5 excluded")

#remove outliers
Data_trp=Data_trp.where(Data_trp['temperature']>=min_T).where(Data_trp['temperature']<=max_T)
Data_met=Data_met.where(Data_met['temperature']>=min_T).where(Data_met['temperature']<=max_T)

#extract coords
height=Data_met.height.values
time=Data_met.time.values

#extract data
T_trp=Data_trp.temperature
r_trp=Data_trp.waterVapor/1000
T_met=Data_met.temperature

#calculate differences
df_dz_trp=xr.Dataset()
df_dz_met=xr.Dataset()
for i_h1 in range(len(height)):
    for i_h2 in range(i_h1+1,len(height)):
        df_dz_trp[f'{height[i_h2]}-{height[i_h1]}']=(T_trp.isel(height=i_h2)-T_trp.isel(height=i_h1))/(height[i_h2]-height[i_h1])
        df_dz_met[f'{height[i_h2]}-{height[i_h1]}']=(T_met.isel(height=i_h2)-T_met.isel(height=i_h1))/(height[i_h2]-height[i_h1])

#kinematic
Data_met['um']=Data_met['ws']*np.cos(np.radians(270-Data_met['wd']))
Data_met['vm']=Data_met['ws']*np.sin(np.radians(270-Data_met['wd']))
dum_dz=(Data_met['um'].sel(height=h2)-Data_met['um'].sel(height=h1))/(h2-h1)
dvm_dz=(Data_met['vm'].sel(height=h2)-Data_met['vm'].sel(height=h1))/(h2-h1)

#thermodynamic (met)
e=vapor_pressure(Data_met['dewp_temp'])
P_s=Data_met['press'].isel(height=0)*100
q_s=0.622*e.isel(height=0)/P_s
Tv_s=(T_met.isel(height=0)+273.15)*(1+0.61*q_s)
dP_dz=-config['g']*P_s/(config['R_a']*Tv_s) 
Data_met['press']=(P_s+(Data_met.height-Data_met.height[0])*dP_dz)/100
Data_met['q']=0.622*e/(Data_met['press']*100)
Data_met['Tv']=(T_met+273.15)*(1+0.61*Data_met['q'])
Data_met['theta_v']=Data_met['Tv']*(config['P_ref']/(Data_met['press']*100))**(config['R_a']/config['cp'])

#gradients (met)
theta_v_avg=(Data_met['theta_v'].sel(height=h2)+Data_met['theta_v'].sel(height=h1))/2
dtheta_v_dz=(Data_met['theta_v'].sel(height=h2)-Data_met['theta_v'].sel(height=h1))/(h2-h1)
Ri_met=config['g']/theta_v_avg*dtheta_v_dz/(dum_dz**2+dvm_dz**2)
Ri_met=Ri_met.where((Ri_met>=bin_Ri[0])*(Ri_met<bin_Ri[-1]))

#thermodynamic (TROPoe)
Data_trp['q']=1/(1/r_trp+1)
Data_trp['Tv']=(T_trp+273.15)*(1+0.61*Data_trp['q'])
dP_dz=-config['g']*P_s/(config['R_a']*Data_trp['Tv'].isel(height=0)) 
Data_trp['press']=(P_s+(Data_trp.height-Data_trp.height[0])*dP_dz)/100
Data_trp['theta_v']= Data_trp['Tv']*(config['P_ref']/(Data_trp['press']*100))**(config['R_a']/config['cp'])
theta_v_avg=(Data_trp['theta_v'].sel(height=h2)+Data_trp['theta_v'].sel(height=h1))/2
dtheta_v_dz=(Data_trp['theta_v'].sel(height=h2)-Data_trp['theta_v'].sel(height=h1))/(h2-h1)
Ri_trp=config['g']/theta_v_avg*dtheta_v_dz/(dum_dz**2+dvm_dz**2)
Ri_trp=Ri_trp.where((Ri_trp>=bin_Ri[0])*(Ri_trp<bin_Ri[-1]))

#compare stability
real=~np.isnan(Ri_met+Ri_trp)

Ri_met=Ri_met.where(real)
Ri_trp=Ri_trp.where(real)

stab_check=np.zeros((len(stab_names),len(stab_names)))
i_met=0
for s_met in stab_names:
    i_Ri_met=stab_names[s_met]
    sel_met=(Ri_met>=bin_Ri[i_Ri_met])*(Ri_met<bin_Ri[i_Ri_met+1])
    i_trp=0
    for s_trp in stab_names:
        i_Ri_trp=stab_names[s_trp]
        sel_trp=(Ri_trp>=bin_Ri[i_Ri_trp])*(Ri_trp<bin_Ri[i_Ri_trp+1])
        stab_check[i_met,i_trp]=np.sum(sel_met*sel_trp)/np.sum(sel_met)*100
        i_trp+=1
    i_met+=1

#%% Plots

#linear regression
bins=np.arange(-5,5.1,0.05)
fig=plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2,3+1,width_ratios=[1,1,1,0.05]) 
ctr=0
for v in df_dz_trp.data_vars:
    ax=fig.add_subplot(gs[int(ctr/3),ctr-int(ctr/3)*3])
    if ctr==2 or ctr==5:
        cax=fig.add_subplot(gs[int(ctr/3),ctr-int(ctr/3)*3+1])
    else:
        cax=None
    
    utl.plot_lin_fit(df_dz_met[v].values,df_dz_trp[v].values,ax=ax,cax=cax,bins=50,legend=ctr==0,limits=[0,100])
    
    ax.grid(True)
    if ctr>=3:
        ax.set_xlabel(r'$\frac{\Delta T}{\Delta z}$ (met) [$^\circ$C m$^{-1}$]')
    if ctr==0 or ctr==3:
        ax.set_ylabel(r'$\frac{\Delta T}{\Delta z}$ (tropoe) [$^\circ$C m$^{-1}$]')
        
    if ctr==0:
        plt.legend(draggable=True)
    
    vmax=np.max([np.abs(df_dz_met[v]).max(),np.abs(df_dz_trp[v]).max()])
    ax.set_xlim([-vmax,vmax])
    ax.set_ylim([-vmax,vmax])
    
    ticks=np.arange(-np.round(vmax/0.05)*0.05,np.round(vmax/0.05)*0.05+0.01,0.05)
    ax.set_aspect('equal')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
    
    inset_ax = ax.inset_axes([0.8, 0.1, 0.1, 0.5])
    inset_ax.plot([0,0],[0,135],'k',linewidth=2)
    inset_ax.plot([0,0.1],[int(v.split('-')[0]),int(v.split('-')[0])],'-k')
    inset_ax.plot(0.1,int(v.split('-')[0]),'.k',markersize=10)
    inset_ax.plot([0,0.1],[int(v.split('-')[1]),int(v.split('-')[1])],'-k')
    inset_ax.plot(0.1,int(v.split('-')[1]),'.k',markersize=10)
    inset_ax.set_xlim([-0.1,0.3])
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    ctr+=1
        
matrix = to_latex(np.int32(stab_check))
print(matrix)

#Ri time series
plt.figure(figsize=(18,4))

plt.plot(time,Ri_met,'-k',label='Met')
plt.plot(Data_met_sta.time,Data_met_sta[f'Ri_{h1}_{h2}'],'b',label='Met (10 min)')
plt.plot(time,Ri_trp,'-r',label='TROPoe')
plt.gca().set_yscale('symlog')
plt.ylim([-100,100])
plt.grid(True)
plt.xlabel('Time (UTC)')
plt.ylabel('Ri')
plt.legend()

#Stability check



