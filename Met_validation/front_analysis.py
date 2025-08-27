# -*- coding: utf-8 -*-
"""
Plot data during a frontal passage
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
import matplotlib
from matplotlib import pyplot as plt
import yaml
import matplotlib.dates as mdates
import glob
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi']=500
plt.close("all")

#%% Inputs
path_config=os.path.join(cd,'configs/config.yaml') #config path
source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')#layout

sites_trp=['B','C1a','G']
sites_met=['A1','A2','A5','A7','B','C1a','G']

sources_trp={'B':'sb.assist.tropoe.z01.c0/sb.assist.tropoe.z01.c0.20230805.001005.nc',
             'C1a':'sc1.assist.tropoe.z01.c0/sc1.assist.tropoe.z01.c0.20230805.001005.nc',
             'G':'sg.assist.tropoe.z01.c0/sg.assist.tropoe.z01.c0.20230805.001005.nc'}
sources_met={'A1':'sa1.met.z01.b0.20230805*nc',
             'A2':'sa2.met.z01.b0.20230805*nc',
             'A5':'sa5.met.z01.b0.20230805*nc',
             'A7':'sa7.met.z01.b0.20230805*nc',
             'B':'sb.met.z01.b0.20230805*nc',
             'C1a':'sc1.met.z01.b0.20230805*nc',
             'G':'sg.met.z01.b0.20230805*nc',}

path_trp='C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken'
path_met=os.path.join(cd,'data','front')

times=np.array([np.datetime64('2023-08-05T10:41:09'),
       np.datetime64('2023-08-05T10:51:25'),
       np.datetime64('2023-08-05T10:59:49'),
       np.datetime64('2023-08-05T11:09:58'),
       np.datetime64('2023-08-05T11:18:22'),
       np.datetime64('2023-08-05T11:31:00')])

farms_sel=['Armadillo Flats','King Plains','unknown Garfield County','Breckinridge']

#QC
max_gamma=3 #maximumm gamma in TROPoe

#graphics
T_min=24
T_max=28

#%% Functions
def three_point_star():
    # Points of a 3-pointed star (scaled and centered)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points (3 outer, 3 inner)
    outer_radius = 1
    inner_radius = 0.1
    coords = []

    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        coords.append((x, y))

    coords.append(coords[0])  # close the shape
    return Path(coords)

#%% Initialization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#read layout
Map=xr.open_dataset(source_layout,group='ground_sites')
Turbines=xr.open_dataset(source_layout,group='turbines').rename({'Wind plant':'wind_plant'})

#read TROPoe
T_trp={}
for s in sites_trp:
    
    Data_trp=xr.open_dataset(os.path.join(path_trp,sources_trp[s]))
    
    #qc tropoe data
    Data_trp['cbh'][(Data_trp['lwp']<config['min_lwp']).compute()]=Data_trp['height'].max()#remove clouds with low lwp
    
    qc_gamma=Data_trp['gamma']<=max_gamma
    qc_rmsa=Data_trp['rmsa']<=config['max_rmsa']
    qc_cbh=Data_trp['height']<Data_trp['cbh']
    qc=qc_gamma*qc_rmsa*qc_cbh
    Data_trp['qc']=~qc+0
        
    print(f'{np.round(np.sum(qc_gamma).values/qc_gamma.size*100,1)}% retained after gamma filter', flush=True)
    print(f'{np.round(np.sum(qc_rmsa).values/qc_rmsa.size*100,1)}% retained after rmsa filter', flush=True)
    print(f'{np.round(np.sum(qc_cbh).values/qc_cbh.size*100,1)}% retained after cbh filter', flush=True)
    
    #fix height
    Data_trp=Data_trp.assign_coords(height=Data_trp.height*1000+config['height_assist'])
     
    T_trp[s]=Data_trp.temperature.where(Data_trp.qc==0)
    
    Data_trp.close()

#read met
T_met={}
for s in sites_met:
    files=glob.glob(os.path.join(path_met,sources_met[s]))
    Data_met=xr.open_mfdataset(files)
    
    T_met[s]=Data_met.temperature.where(Data_met.qc_temperature==0)
    
    Data_met.close()
    
#%% Main

#interpolate in time/height
T_trp_int={}
for s in sites_trp:
    T_trp_int[s]=T_trp[s].interp(height=2,time=times)
    
T_met_int={}
for s in sites_met:
    T_met_int[s]=T_met[s].interp(time=times)

#%% Plots
plt.close("all")
star_marker = MarkerStyle(three_point_star())
fig=plt.figure(figsize=(25,4))
gs = gridspec.GridSpec(1,len(times)+1,width_ratios=[1]*len(times)+[0.1]) 
for i in range(len(times)):
    ax=fig.add_subplot(gs[i])
    
    for wf in farms_sel:
        x_turbine=Turbines.x_utm.where(Turbines.wind_plant==wf).values-Map.x_utm.sel(site='C1a').values
        y_turbine=Turbines.y_utm.where(Turbines.wind_plant==wf).values-Map.y_utm.sel(site='C1a').values
        for xt,yt in zip(x_turbine,y_turbine):
            plt.plot(xt,yt,'xk', marker=star_marker, markersize=5, color='k')
            
    for s in sites_met:
        sc=plt.scatter(Map.x_utm.sel(site=s)-Map.x_utm.sel(site='C1a'),Map.y_utm.sel(site=s)-Map.y_utm.sel(site='C1a'),
                    c=T_met_int[s].isel(time=i),s=100,edgecolor='k',cmap='hot',vmin=T_min,vmax=T_max,zorder=10)
        
    plt.xlabel('W-E [m]')
    if i>0:
        ax.set_yticklabels([])
    else:
        plt.ylabel('S-N [m]')
    plt.xlim([-15000,15000])
    plt.ylim([-16000,14000])
    plt.grid()
    ax.set_aspect('equal')
    plt.title(str(times[i]).split('T')[-1]+' UTC')
    
cax=fig.add_subplot(gs[0,i+1])
plt.colorbar(sc,cax,label=r'$T$ [$^\circ$C]')

matplotlib.rcParams['font.size'] = 16
fig=plt.figure(figsize=(18,8))
gs = gridspec.GridSpec(len(sites_trp),2,width_ratios=[1,0.03]) 
ctr=0
for s in sites_trp:
    ax=fig.add_subplot(gs[ctr,0])
    cf=plt.contourf(T_trp[s].time,T_trp[s].height,T_trp[s].T,np.arange(T_min,T_max+.1,0.2),cmap='hot',extend='both')
    plt.contour(T_trp[s].time,T_trp[s].height,T_trp[s].T,np.arange(T_min,T_max+.1,0.2),colors='k',linewidths=1,alpha=0.25,extend='both')
    
    if ctr==len(sites_trp)-1:
        plt.xlabel('Time (UTC)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax.set_xticklabels([])
    plt.ylabel(r'$z$ [m]')
    plt.grid()
    sc=plt.scatter(T_met_int[s].time,np.zeros(len(times))+2,
                c=T_met_int[s],s=50,edgecolor='k',cmap='hot',vmin=T_min,vmax=T_max,zorder=10)
    
    plt.xlim([times[0]-np.timedelta64(600,'s'),times[-1]+np.timedelta64(600,'s')])
    plt.ylim([-20,500])
    
    ctr+=1

cax = fig.add_subplot(gs[:, 1])
plt.colorbar(cf,cax,label=r'$T$ [$^\circ$C]')
cax.set_yticks(np.arange(T_min,T_max+1))

plt.figure(figsize=(18,6))
for s in sites_met:
    plt.plot(T_met[s].time,T_met[s],label=s)
plt.ylabel(r'$T$ [$^\circ$C]')
plt.xlabel('Time (UTC)')
plt.legend()
plt.grid()