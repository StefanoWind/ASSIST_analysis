# -*- coding: utf-8 -*-
"""
Identify periods with wakes
"""
import os
cd=os.path.dirname(__file__)
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import warnings
import matplotlib
import glob
import utm
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_turbines={'GE1.5':  'data/nwtc/scada/ge1.5_v2.csv',
                 'SGRE2.3':'data/nwtc/scada/sgre2.3.csv',
                 'SGRE2.0':'data/nwtc/scada/sgre2.0_v2.csv'}#sources of SCADA power

source_met=os.path.join(cd,'data/nwtc/nwtc.m5.c1/*nc')#source of met stats

source_layout=os.path.join(cd,'data','NWTC.xlsx')#source of site layout

#site info
timezone = 'America/Denver'#timezone
turbines=['GE1.5','SGRE2.3','SGRE2.0']#turbine names
sites=['Site 3.2','M5','Site 4.0','M2']#wakes sites
height_sel=87#[m] selected height fo wind direction

#stats
min_power=0.1#minimum normalized power

#graphics
colors={'GE1.5':  'g',
        'SGRE2.3':'r',
        'SGRE2.0':'b'}

#%% Functions
def angle_diff(a1, a2):
    diff = a1-a2
    return (diff+180)%360-180

#%% Initialization

#read site layout
FC=pd.read_excel(source_layout).set_index('Site')

#locations
xy=utm.from_latlon(FC['Lat'].values,FC['Lon'].values)
xref=utm.from_latlon(FC['Lat']['M5'],FC['Lon']['M5'])[0]
yref=utm.from_latlon(FC['Lat']['M5'],FC['Lon']['M5'])[1]
FC['x']=xy[0]-xref
FC['y']=xy[1]-yref

#read met data
files=glob.glob(source_met)
met=xr.open_mfdataset(files)
wd=met.wd.sel(height=height_sel)

#read power data
power=xr.Dataset()
for turbine in turbines:
    power_raw=pd.read_csv(source_turbines[turbine])
    power_time=[np.datetime64(pd.Timestamp(t, tz=timezone).tz_convert("UTC")) for t in power_raw['Timestamp']]
    power_power=[np.max([-np.float64(str(p).replace('kW','')),0]) for p in power_raw.iloc[:,-1].values]
    power0=xr.DataArray(data=power_power,coords={'time':power_time})
    power[turbine]=power0.interp(time=wd.time)

#%% Main

#define waked angles (IEC 61400-12-1 2023)
ang_wake={}
for site in sites:
    xs=FC['x'][site]
    ys=FC['y'][site]
    for turbine in turbines:
        xt=FC['x'][turbine]
        yt=FC['y'][turbine]
        D=FC['Diameter'][turbine]
        L=((xs-xt)**2+(ys-yt)**2)**0.5
        
        if L/D>2:
            ang_wake[f"{site}-{turbine}"]=1.3*np.degrees(np.arctan(2.5*D/L+0.15))+10
        else:
            ang_wake[f"{site}-{turbine}"]=360

waked=xr.Dataset()
for site in sites:
    waked_all=np.zeros((len(wd),len(turbines)))
    
    xs=FC['x'][site]
    ys=FC['y'][site]
    ctr=0
    for turbine in turbines:
        xt=FC['x'][turbine]
        yt=FC['y'][turbine]
        D=FC['Diameter'][turbine]
        power_rated=FC['Power'][turbine]
        heading=(270-np.degrees(np.arctan2(ys-yt,xs-xt)))%360
        
        ang_condition=np.abs(angle_diff(wd,heading))<ang_wake[f"{site}-{turbine}"]/2
        power_condition=power[turbine]/power_rated>min_power
        waked_all[:,ctr]=ang_condition*power_condition
        
        print(f"{np.sum(ang_condition*power_condition).values} waked periods at {site} due to {turbine}")
        ctr+=1
        
    waked[site]=xr.DataArray(data=waked_all,coords={'time':wd.time,'turbine':turbines})
    
#%% Output
waked.to_netcdf(os.path.join(cd,'data/turbine_wakes.nc'))

#%% Plots
plt.figure(figsize=(16,10))
for site in sites:
    xs=FC['x'][site]
    ys=FC['y'][site]
    plt.plot(xs,ys,'xk')
    ctr=1
    for turbine in turbines:
        sel=waked[site].sel(turbine=turbine)==1
        plt.plot(xs+np.cos(np.radians(90-wd[sel]))*50*ctr,ys+np.sin(np.radians(90-wd[sel]))*50*ctr,'.',alpha=0.25,color=colors[turbine])
        
        ctr+=1

for turbine in turbines:
    plt.plot(FC['x'][turbine],FC['y'][turbine],'*',color=colors[turbine])
plt.gca().set_aspect("equal")
plt.grid()
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')